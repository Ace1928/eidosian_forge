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
def varcorrection_unequal(var_all, nobs_all, df_all):
    """return joint variance from samples with unequal variances and unequal
    sample sizes

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
    varjoint : float
        joint variance.
    dfjoint : float
        joint Satterthwait's degrees of freedom


    Notes
    -----
    (copy, paste not correct)
    variance is

    1/k * sum_i 1/n_i

    where k is the number of samples and summation is over i=0,...,k-1.
    If all n_i are the same, then the correction factor is 1/n.

    This needs to be multiplies by the joint variance estimate, means square
    error, MSE. To obtain the correction factor for the standard deviation,
    square root needs to be taken.

    This is for variance of mean difference not of studentized range.
    """
    var_all = np.asarray(var_all)
    var_over_n = var_all * 1.0 / nobs_all
    varjoint = var_over_n.sum()
    dfjoint = varjoint ** 2 / (var_over_n ** 2 * df_all).sum()
    return (varjoint, dfjoint)