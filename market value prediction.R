setwd("~/Desktop/MSBA/452 Mahchine Learning/Class 3/")
df <- read.csv('FIFA_Player_List.csv')
#install.packages('funModeling')
#install.packages('corrplot')
#install.packages('mlbench')

library(dplyr)
library(tidyverse)
library(ggplot2)
library(Hmisc)
library(funModeling) 
library(corrplot)
library(caret)
library(mlbench)
library(glmnet)
library(forecast)
set.seed(86)
################################################################
#####Section 1 Data Analysis ###################################
################################################################
summary(df)

basic_eda <- function(data)
{
  ## Look the head of the dataset
  glimpse(data)
  
  ## Check missing values and check how many unique values for each variables 
  print(status(data))
  
  ## Know the frequency for categorical data
  print(profiling_num(data))
  
  ## Plot histograms for each numeric variable and check normality 
  plot_num(data)
  
  ## Check outliers and distributions 
  describe(data)
}

basic_eda(df)

## Check correlation among variables 
df_num <- df[,sapply(df,is.numeric)]
corrplot(cor(df_num), method = 'number', is.corr = FALSE)

################################################################
#####Section 2 Feature Selection ###############################
################################################################
## Transform skewed variables using log  
## Use log for positive skewness
df$Market.Value <- log(df$Market.Value)
df$Weekly.Salary <- log(df$Weekly.Salary)
hist(df$Market.Value)

## Use log(max(x+1) - x) for negative skewness
df$Ball.Skills <- log(max(df$Ball.Skills + 1) - df$Ball.Skills)
df$Physical <- log(max(df$Physical + 1) - df$Physical)
describe(df)

hist(df$Weekly.Salary)
hist(df$Physical)
## Create dummy variables for categorical variable 
df$foot_dummy <- ifelse(df$Preferred.Foot == 'Left', 1, 0)
df$goalkeep_dummy <- ifelse(df$Goalkeeping <= 30, 0, 1 )

## Drop categorical variable 'Preferred foot' 
df_dummy <- dplyr::select(df, -'Preferred.Foot')

#y <- df_dummy['Market.Value']
df_reorder <- dplyr::select(df_dummy, Market.Value, everything())

## Drop 'Player' 
df_reorder <- dplyr::select(df_reorder, -'Player') 

## Split the dataset into training data and testing data 
index_all <- sample(1:nrow(df_reorder), 0.7*nrow(df_reorder))

train_all <- df_reorder[index_all,]
test_all <- df_reorder[-index_all,]

full_model <- lm(Market.Value ~ Overall.Score+Potential.Score+Weekly.Salary + Height + Weight
                 + Age + Ball.Skills + Defence + Mental + Passing + Physical + 
                   Shooting + Goalkeeping + foot_dummy + goalkeep_dummy, data = train_all)
summary(full_model)


## R_squared: 0.96

## Create the evaluation metrics function
eval = function(model, df, predictions, dependent){
  resids = df[,dependent] - predictions 
  resids2 = resids ** 2
  N = length(predictions)
  r2 = as.character(round(summary(model)$r.squared, 2))
  adj_r2 = as.character(round(summary(model)$adj.r.squared,2))
  print(adj_r2) 
  print(as.character(round(sqrt(sum(resids2)/N) , 2))) ## RMSE
}

## Prediction on train data using full-linear regression model
predict_full = predict(full_model, newdata = train_all)
eval(full_model, train_all, predict_full, dependent = 'Market.Value' )

## Prediction on test data using full model
predict_full = predict(full_model, newdata = test_all)
eval(full_model, test_all, predict_full, dependent = 'Market.Value' )


## Learning Vector Quantization Model to estimate the variable importance 
control <- rfeControl(functions = lmFuncs, number = 10)
results <- rfe(df_reorder[,2:15], df_reorder[,1], sizes = c(1:14), 
               metric = 'Rsquared', rfeControl = control )
print(results)

## List the chosen features
predictors(results)

## Plot the results 
plot(results, type = c('g','o'))

reduced_model <- lm(Market.Value ~ Weekly.Salary + Overall.Score + Age + foot_dummy + Goalkeeping, 
                    data = train_all)
predict_reduced = predict(reduced_model, newdata = test_all)
eval(reduced_model, test_all, predict_reduced, dependent = 'Market.Value' )

################################################################
#####Section 3 Model Selection #################################
################################################################
## Data Partitioning

new_df <- df_reorder[,c('Market.Value','Overall.Score', 'Age', 'foot_dummy', 'Mental', 'Ball.Skills')]
set.seed(100)

index = sample(1:nrow(new_df), 0.7 * nrow(new_df))

train = new_df[index,] # Training data
test = new_df[-index,] # Testing data 

dim(train)
dim(test)

## PreProcessing data by scaling and centering without dummy variable
cols = c('Overall.Score', 'Age', 'Mental', 'Ball.Skills')
preProValues <- preProcess(train[,cols], method = c('center', 'scale'))

train[,cols] <- predict(preProValues, train[,cols])
test[,cols] <- predict(preProValues, test[,cols])

summary(test)
summary(train)


##################################################################
##### Ridge Regression ###########################################
##################################################################
x = train %>% select(Overall.Score, Age,Mental,foot_dummy, Ball.Skills) %>% data.matrix()
y_train = train$Market.Value
#y <- train %>% select(Market.Value) %>% scale(center = TRUE , scale = FALSE) %>% as.matrix()

lambdas <- 10^seq(3,-6,length.out = 100)

x_test <- test %>% select(Overall.Score, Age,Mental,foot_dummy, Ball.Skills) %>% data.matrix()
y_test <- test$Market.Value
#x = as.matrix(train_all[,-1])
#y_train <- train_all$Market.Value 
ridge_reg <- glmnet(x,y_train, alpha = 0, lambda = lambdas)
summary(ridge_reg)
plot(ridge_reg,xvar = 'lambda', label = TRUE)

cv_fit <- cv.glmnet(x,y_train,alpha = 0, lambda = lambdas)
plot(cv_fit)

## Get the optimal lambda
opt_lambda <- cv_fit$lambda.min
opt_lambda

fit <- cv_fit$glmnet.fit 
summary(fit)

## Use function to compute R^2 for both true and predicted values
eval2 <- function(true, predicted, df) {
  SSE <- sum((predicted - true) ^ 2) 
  SST <- sum((true - mean(true)) ^ 2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
}

## Prediction and evaluation on training data
predict_train <- predict(cv_fit, s = opt_lambda, newx = x)
eval2(y_train, predict_train,train)
resid_train <- resid(cv_fit, s = opt_lambda, newx = x)

## Prediction and evaluation on testing data
predict_test <- predict(cv_fit, s = opt_lambda, newx = x_test)
eval2(y_test, predict_test, test)
resid_test <- resid(cv_fit, s = opt_lambda, newx = x_test)
##################################################################
##### Lasso Regression ###########################################
##################################################################
lasso_reg <- cv.glmnet(x, y_train, alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = 5)

plot(lasso_reg)

lambda_best <- lasso_reg$lambda.min
lambda_best

lasso_model <- glmnet(x, y_train, alpha = 1, lambda = lambda_best, standardized = TRUE)
predictions_train <- predict(lasso_model, s = lambda_best, newx = x)
residual_train <- resid(lasso_model, s = lambda_best, newx = x)
eval2(y_train, predictions_train, train)

## Make prediction on testing data
x_test = test %>% select(Overall.Score, Age,Mental,foot_dummy, Ball.Skills) %>% data.matrix()
y_test = test$Market.Value

predictions_test <- predict(lasso_model, s = lambda_best, newx = x_test)
residual_test <- resid(lasso_model, s = lambda_best, newx = x_test)
eval2(y_test, predictions_test, test)

eval <- list(eval2(y_train, predictions_train, train), eval2(y_test, predict_test, test), eval2(y_train, predictions_train, train), eval2(y_test, predictions_test, test))
names(eval) <- c('ridge train', 'ridge test', 'lasso train', 'lasso train')

################################################################
#####Section 4 Model Evaluation ################################
################################################################
#Divide the data into test and train and evaluate the model along with fit, residuals, and standard error plots.

#Fit
#ridge
plot(fit,xvar = "lambda", label = TRUE)
plot(fit,xvar = "dev", label = TRUE)

#residual
#ridge
plot(y_test,resid_test)

#Standard error
#ridge
plot(predict_test, (predict_test-y_test)^2, ylab = "SE")

################################################################
#####Section 5 Conclusion ######################################
################################################################

#see report