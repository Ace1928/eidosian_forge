from statsmodels.compat.python import lzip
from typing import Callable
import numpy as np
import pandas as pd
from scipy import optimize, stats
from statsmodels.stats.base import AllPairsResults, HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.tools.validation import array_like
def _power_ztost(mean_low, var_low, mean_upp, var_upp, mean_alt, var_alt, alpha=0.05, discrete=True, dist='norm', nobs=None, continuity=0, critval_continuity=0):
    """
    Generic statistical power function for normal based equivalence test

    This includes options to adjust the normal approximation and can use
    the binomial to evaluate the probability of the rejection region

    see power_ztost_prob for a description of the options
    """
    if not isinstance(continuity, tuple):
        continuity = (continuity, continuity)
    crit = stats.norm.isf(alpha)
    k_low = mean_low + np.sqrt(var_low) * crit
    k_upp = mean_upp - np.sqrt(var_upp) * crit
    if discrete or dist == 'binom':
        k_low = np.ceil(k_low * nobs + 0.5 * critval_continuity)
        k_upp = np.trunc(k_upp * nobs - 0.5 * critval_continuity)
        if dist == 'norm':
            k_low = k_low * 1.0 / nobs
            k_upp = k_upp * 1.0 / nobs
    if np.any(k_low > k_upp):
        import warnings
        warnings.warn('no overlap, power is zero', HypothesisTestWarning)
    std_alt = np.sqrt(var_alt)
    z_low = (k_low - mean_alt - continuity[0] * 0.5 / nobs) / std_alt
    z_upp = (k_upp - mean_alt + continuity[1] * 0.5 / nobs) / std_alt
    if dist == 'norm':
        power = stats.norm.cdf(z_upp) - stats.norm.cdf(z_low)
    elif dist == 'binom':
        power = stats.binom.cdf(k_upp, nobs, mean_alt) - stats.binom.cdf(k_low - 1, nobs, mean_alt)
    return (power, (k_low, k_upp, z_low, z_upp))