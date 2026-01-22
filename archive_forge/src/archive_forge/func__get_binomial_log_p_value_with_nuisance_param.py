from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
def _get_binomial_log_p_value_with_nuisance_param(nuisance_param, x1_sum_x2, x1_sum_x2_log_comb, index_arr):
    """
    Compute the log pvalue in respect of a nuisance parameter considering
    a 2x2 sample space.

    Parameters
    ----------
    nuisance_param : float
        nuisance parameter used in the computation of the maximisation of
        the p-value. Must be between 0 and 1

    x1_sum_x2 : ndarray
        Sum of x1 and x2 inside barnard_exact

    x1_sum_x2_log_comb : ndarray
        sum of the log combination of x1 and x2

    index_arr : ndarray of boolean

    Returns
    -------
    p_value : float
        Return the maximum p-value considering every nuisance parameter
        between 0 and 1

    Notes
    -----

    Both Barnard's test and Boschloo's test iterate over a nuisance parameter
    :math:`\\pi \\in [0, 1]` to find the maximum p-value. To search this
    maxima, this function return the negative log pvalue with respect to the
    nuisance parameter passed in params. This negative log p-value is then
    used in `shgo` to find the minimum negative pvalue which is our maximum
    pvalue.

    Also, to compute the different combination used in the
    p-values' computation formula, this function uses `gammaln` which is
    more tolerant for large value than `scipy.special.comb`. `gammaln` gives
    a log combination. For the little precision loss, performances are
    improved a lot.
    """
    t1, t2 = x1_sum_x2.shape
    n = t1 + t2 - 2
    with np.errstate(divide='ignore', invalid='ignore'):
        log_nuisance = np.log(nuisance_param, out=np.zeros_like(nuisance_param), where=nuisance_param >= 0)
        log_1_minus_nuisance = np.log(1 - nuisance_param, out=np.zeros_like(nuisance_param), where=1 - nuisance_param >= 0)
        nuisance_power_x1_x2 = log_nuisance * x1_sum_x2
        nuisance_power_x1_x2[(x1_sum_x2 == 0)[:, :]] = 0
        nuisance_power_n_minus_x1_x2 = log_1_minus_nuisance * (n - x1_sum_x2)
        nuisance_power_n_minus_x1_x2[(x1_sum_x2 == n)[:, :]] = 0
        tmp_log_values_arr = x1_sum_x2_log_comb + nuisance_power_x1_x2 + nuisance_power_n_minus_x1_x2
    tmp_values_from_index = tmp_log_values_arr[index_arr]
    max_value = tmp_values_from_index.max()
    with np.errstate(divide='ignore', invalid='ignore'):
        log_probs = np.exp(tmp_values_from_index - max_value).sum()
        log_pvalue = max_value + np.log(log_probs, out=np.full_like(log_probs, -np.inf), where=log_probs > 0)
    return -log_pvalue