from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
@deprecate_kwarg('results', 'res')
def acorr_breusch_godfrey(res, nlags=None, store=False):
    """
    Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.

    Parameters
    ----------
    res : RegressionResults
        Estimation results for which the residuals are tested for serial
        correlation.
    nlags : int, optional
        Number of lags to include in the auxiliary regression. (nlags is
        highest lag).
    store : bool, default False
        If store is true, then an additional class instance that contains
        intermediate results is returned.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The value of the f statistic for F test, alternative version of the
        same test based on F test for the parameter restriction.
    fpval : float
        The pvalue for F test.
    res_store : ResultsStore
        A class instance that holds intermediate results. Only returned if
        store=True.

    Notes
    -----
    BG adds lags of residual to exog in the design matrix for the auxiliary
    regression with residuals as endog. See [1]_, section 12.7.1.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
      5th edition. (2002).
    """
    x = np.asarray(res.resid).squeeze()
    if x.ndim != 1:
        raise ValueError('Model resid must be a 1d array. Cannot be used on multivariate models.')
    exog_old = res.model.exog
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(10, nobs // 5)
    x = np.concatenate((np.zeros(nlags), x))
    xdall = lagmat(x[:, None], nlags, trim='both')
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = x[-nobs:]
    if exog_old is None:
        exog = xdall
    else:
        exog = np.column_stack((exog_old, xdall))
    k_vars = exog.shape[1]
    resols = OLS(xshort, exog).fit()
    ft = resols.f_test(np.eye(nlags, k_vars, k_vars - nlags))
    fval = ft.fvalue
    fpval = ft.pvalue
    fval = float(np.squeeze(fval))
    fpval = float(np.squeeze(fpval))
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, nlags)
    if store:
        res_store = ResultsStore()
        res_store.resols = resols
        res_store.usedlag = nlags
        return (lm, lmpval, fval, fpval, res_store)
    else:
        return (lm, lmpval, fval, fpval)