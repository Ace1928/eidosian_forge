import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def _var_cmle_negbin(rate1, rate2, nobs_ratio, exposure=1, value=1, dispersion=0):
    """
    variance based on constrained cmle, for score test version

    for ratio comparison of two negative binomial samples

    value = rate1 / rate2 under the null
    """
    rate0 = rate2
    nobs_ratio = 1 / nobs_ratio
    a = -dispersion * exposure * value * (1 + nobs_ratio)
    b = dispersion * exposure * (rate0 * value + nobs_ratio * rate1) - (1 + nobs_ratio * value)
    c = rate0 + nobs_ratio * rate1
    if dispersion == 0:
        r0 = -c / b
    else:
        r0 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    r1 = r0 * value
    v = 1 / exposure / r0 * (1 + 1 / value / nobs_ratio) + (1 + nobs_ratio) / nobs_ratio * dispersion
    r2 = r0
    return (v * nobs_ratio, r1, r2)