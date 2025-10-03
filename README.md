# Mortgage Loan Interest Rate Analysis

## Project Overview
This project analyzes determinants of **mortgage loan interest rates** using a 2011 sample from the **Freddie Mac Single-Family Loan-Level Dataset**.  
I combined **econometric methods** (OLS, semi-log specifications, state fixed effects) with **machine learning models** (Decision Tree, Random Forest, SVR) to compare interpretability and predictive accuracy.

The goal: understand which borrower and loan characteristics explain interest rate variation, and benchmark prediction performance across approaches.

## Key Questions
1. Do borrower risk factors (Credit Score, DTI, LTV) significantly affect mortgage interest rates?  
2. How much variation can be explained by loan/borrower characteristics alone?  
3. Do geographic differences (state fixed effects) matter?  
4. Can ML models predict rates more accurately than econometric regressions?  


## Dataset
- **Source:** Freddie Mac Single-Family Loan-Level Dataset (public sample, 2011 originations)  
- **Size:** Hundreds of thousands of loans (sample filtered for feasibility)  
- **Key Variables:**  
  - Credit Score, DTI, LTV, UPB (loan balance), Interest Rate, Loan Term, State  
- **Data Cleaning:**  
  - Removed placeholders (DTI=999, LTV=999, Credit_Score=9999)  
  - Created engineered features:  
    - `UPB_Thousands` (loan balance scaled)  
    - `log_Credit_Score`, `log_UPB`  
    - Loan term bins (≤10, 11–15, 16–20, 21–25, 26–30 years)  


## Methods
- **Econometrics (statsmodels):**  
  - OLS with HC1 robust standard errors  
  - Joint significance test (F-test)  
  - Semi-log regression specification  
  - State fixed effects  
- **Machine Learning (scikit-learn):**  
  - Decision Tree Regressor (tuned with GridSearchCV)  
  - Random Forest Regressor (tuned with GridSearchCV)  
  - Support Vector Regression (RBF kernel, features scaled with StandardScaler)  
- **Evaluation Metrics:** R², RMSE on hold-out test set  


## Results

### Econometric Models
- **OLS (robust):** R² ≈ **0.11**; all predictors (Credit Score, DTI, LTV, UPB) statistically significant  
- **Semi-log OLS:** Slightly better fit (R² ≈ **0.113**); elasticities more stable  
- **Fixed Effects:** Adding state dummies raised R² ≈ **0.148**, showing meaningful geographic heterogeneity  
- **Key coefficients:**  
  - +1 point DTI ≈ +0.0075 pp in rate  
  - +1 point LTV ≈ +0.0074 pp in rate  
  - +1 point Credit Score ≈ −0.002 pp in rate  

### ML Benchmark
- **Decision Tree (tuned):** RMSE ≈ **0.60**, R² ≈ **0.12**  
- **SVR (tuned, scaled):** RMSE ≈ **0.53**, R² ≈ **0.12**  
- **Random Forest (tuned):** RMSE ≈ **0.53**, R² ≈ **0.136** (best performer)  

Interpretation: ML improves prediction slightly (lower RMSE, higher R²) but does not offer clear interpretability like regression coefficients.


Credits: David Chen, Bamboo Shen
