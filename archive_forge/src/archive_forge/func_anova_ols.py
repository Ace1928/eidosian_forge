from statsmodels.compat.python import lmap
import os
import numpy as np
from scipy import stats
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from .try_ols_anova import data2dummy
def anova_ols(y, x):
    X = add_constant(data2dummy(x), prepend=False)
    res = OLS(y, X).fit()
    return (res.fvalue, res.f_pvalue, res.rsquared, np.sqrt(res.mse_resid))