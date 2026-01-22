from collections import defaultdict
import numpy as np
from numpy import hstack, vstack
from numpy.linalg import inv, svd
import scipy
import scipy.stats
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.validation import string_like
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.coint_tables import c_sja, c_sjt
from statsmodels.tsa.tsatools import duplication_matrix, lagmat, vec
from statsmodels.tsa.vector_ar.hypothesis_test_results import (
import statsmodels.tsa.vector_ar.irf as irf
import statsmodels.tsa.vector_ar.plotting as plot
from statsmodels.tsa.vector_ar.util import get_index, seasonal_dummies
from statsmodels.tsa.vector_ar.var_model import (
def _linear_trend(nobs, k_ar, coint=False):
    """
    Construct an ndarray representing a linear trend in a VECM.

    Parameters
    ----------
    nobs : int
        Number of observations excluding the presample.
    k_ar : int
        Number of lags in levels.
    coint : bool, default: False
        If True (False), the returned array represents a linear trend inside
        (outside) the cointegration relation.

    Returns
    -------
    ret : ndarray (nobs)
        An ndarray representing a linear trend in a VECM

    Notes
    -----
    The returned array's size is nobs and not nobs_tot so it cannot be used to
    construct the exog-argument of VECM's __init__ method.
    """
    ret = np.arange(nobs) + k_ar
    if not coint:
        ret += 1
    return ret