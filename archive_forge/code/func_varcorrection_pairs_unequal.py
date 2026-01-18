from pystatsmodels mailinglist 20100524
from collections import namedtuple
from statsmodels.compat.python import lzip, lrange
import copy
import math
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from scipy import stats, interpolate
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.multitest import multipletests, _ecdf as ecdf, fdrcorrection as fdrcorrection0, fdrcorrection_twostage
from statsmodels.graphics import utils
from statsmodels.tools.sm_exceptions import ValueWarning
def varcorrection_pairs_unequal(var_all, nobs_all, df_all):
    """return joint variance from samples with unequal variances and unequal
    sample sizes for all pairs

    something is wrong

    Parameters
    ----------
    var_all : array_like
        The variance for each sample
    nobs_all : array_like
        The number of observations for each sample
    df_all : array_like
        degrees of freedom for each sample

    Returns
    -------
    varjoint : ndarray
        joint variance.
    dfjoint : ndarray
        joint Satterthwait's degrees of freedom


    Notes
    -----

    (copy, paste not correct)
    variance is

    1/k * sum_i 1/n_i

    where k is the number of samples and summation is over i=0,...,k-1.
    If all n_i are the same, then the correction factor is 1.

    This needs to be multiplies by the joint variance estimate, means square
    error, MSE. To obtain the correction factor for the standard deviation,
    square root needs to be taken.

    TODO: something looks wrong with dfjoint, is formula from SPSS
    """
    v1, v2 = np.meshgrid(var_all, var_all)
    n1, n2 = np.meshgrid(nobs_all, nobs_all)
    df1, df2 = np.meshgrid(df_all, df_all)
    varjoint = v1 / n1 + v2 / n2
    dfjoint = varjoint ** 2 / (df1 * (v1 / n1) ** 2 + df2 * (v2 / n2) ** 2)
    return (varjoint, dfjoint)