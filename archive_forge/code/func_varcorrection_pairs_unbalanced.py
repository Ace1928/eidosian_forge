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
def varcorrection_pairs_unbalanced(nobs_all, srange=False):
    """correction factor for variance with unequal sample sizes for all pairs

    this is just a harmonic mean

    Parameters
    ----------
    nobs_all : array_like
        The number of observations for each sample
    srange : bool
        if true, then the correction is divided by 2 for the variance of
        the studentized range statistic

    Returns
    -------
    correction : ndarray
        Correction factor for variance.


    Notes
    -----

    variance correction factor is

    1/k * sum_i 1/n_i

    where k is the number of samples and summation is over i=0,...,k-1.
    If all n_i are the same, then the correction factor is 1.

    This needs to be multiplies by the joint variance estimate, means square
    error, MSE. To obtain the correction factor for the standard deviation,
    square root needs to be taken.

    For the studentized range statistic, the resulting factor has to be
    divided by 2.

    """
    n1, n2 = np.meshgrid(nobs_all, nobs_all)
    if not srange:
        return 1.0 / n1 + 1.0 / n2
    else:
        return (1.0 / n1 + 1.0 / n2) / 2.0