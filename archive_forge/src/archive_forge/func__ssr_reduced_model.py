from statsmodels.compat.python import lrange
import numpy as np
import pandas as pd
from pandas import DataFrame, Index
import patsy
from scipy import stats
from statsmodels.formula.formulatools import (
from statsmodels.iolib import summary2
from statsmodels.regression.linear_model import OLS
def _ssr_reduced_model(y, x, term_slices, params, keys):
    """
    Residual sum of squares of OLS model excluding factors in `keys`
    Assumes x matrix is orthogonal

    Parameters
    ----------
    y : array_like
        dependent variable
    x : array_like
        independent variables
    term_slices : a dict of slices
        term_slices[key] is a boolean array specifies the parameters
        associated with the factor `key`
    params : ndarray
        OLS solution of y = x * params
    keys : keys for term_slices
        factors to be excluded

    Returns
    -------
    rss : float
        residual sum of squares
    df : int
        degrees of freedom
    """
    ind = _not_slice(term_slices, keys, x.shape[1])
    params1 = params[ind]
    ssr = np.subtract(y, x[:, ind].dot(params1))
    ssr = ssr.T.dot(ssr)
    df_resid = len(y) - len(params1)
    return (ssr, df_resid)