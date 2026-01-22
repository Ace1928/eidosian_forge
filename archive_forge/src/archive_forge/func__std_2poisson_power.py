import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def _std_2poisson_power(rate1, rate2, nobs_ratio=1, alpha=0.05, exposure=1, dispersion=1, value=0, method_var='score'):
    rates_pooled = (rate1 + rate2 * nobs_ratio) / (1 + nobs_ratio)
    if method_var == 'alt':
        v0 = v1 = rate1 + rate2 / nobs_ratio
    else:
        _, r1_cmle, r2_cmle = _score_diff(rate1, 1, rate2 * nobs_ratio, nobs_ratio, value=value, return_cmle=True)
        v1 = rate1 + rate2 / nobs_ratio
        v0 = r1_cmle + r2_cmle / nobs_ratio
    return (rates_pooled, np.sqrt(v0), np.sqrt(v1))