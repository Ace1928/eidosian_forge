import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
def _multivariate_ols_test(hypotheses, fit_results, exog_names, endog_names):

    def fn(L, M, C):
        params, df_resid, inv_cov, sscpr = fit_results
        t1 = L.dot(params).dot(M) - C
        t2 = L.dot(inv_cov).dot(L.T)
        q = matrix_rank(t2)
        H = t1.T.dot(inv(t2)).dot(t1)
        E = M.T.dot(sscpr).dot(M)
        return (E, H, q, df_resid)
    return _multivariate_test(hypotheses, exog_names, endog_names, fn)