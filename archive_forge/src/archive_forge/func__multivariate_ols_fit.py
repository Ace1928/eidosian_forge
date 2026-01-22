import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo
from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
def _multivariate_ols_fit(endog, exog, method='svd', tolerance=1e-08):
    """
    Solve multivariate linear model y = x * params
    where y is dependent variables, x is independent variables

    Parameters
    ----------
    endog : array_like
        each column is a dependent variable
    exog : array_like
        each column is a independent variable
    method : str
        'svd' - Singular value decomposition
        'pinv' - Moore-Penrose pseudoinverse
    tolerance : float, a small positive number
        Tolerance for eigenvalue. Values smaller than tolerance is considered
        zero.
    Returns
    -------
    a tuple of matrices or values necessary for hypotheses testing

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    Notes
    -----
    Status: experimental and incomplete
    """
    y = endog
    x = exog
    nobs, k_endog = y.shape
    nobs1, k_exog = x.shape
    if nobs != nobs1:
        raise ValueError('x(n=%d) and y(n=%d) should have the same number of rows!' % (nobs1, nobs))
    df_resid = nobs - k_exog
    if method == 'pinv':
        pinv_x = pinv(x)
        params = pinv_x.dot(y)
        inv_cov = pinv_x.dot(pinv_x.T)
        if matrix_rank(inv_cov, tol=tolerance) < k_exog:
            raise ValueError('Covariance of x singular!')
        t = x.dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    elif method == 'svd':
        u, s, v = svd(x, 0)
        if (s > tolerance).sum() < len(s):
            raise ValueError('Covariance of x singular!')
        invs = 1.0 / s
        params = v.T.dot(np.diag(invs)).dot(u.T).dot(y)
        inv_cov = v.T.dot(np.diag(np.power(invs, 2))).dot(v)
        t = np.diag(s).dot(v).dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    else:
        raise ValueError('%s is not a supported method!' % method)