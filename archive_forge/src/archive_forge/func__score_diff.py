import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def _score_diff(y1, n1, y2, n2, value=0, return_cmle=False):
    """score test and cmle for difference of 2 independent poisson rates

    """
    count_pooled = y1 + y2
    rate1, rate2 = (y1 / n1, y2 / n2)
    rate_pooled = count_pooled / (n1 + n2)
    dt = rate_pooled - value
    r2_cmle = 0.5 * (dt + np.sqrt(dt ** 2 + 4 * value * y2 / (n1 + n2)))
    r1_cmle = r2_cmle + value
    eps = 1e-20
    v = r1_cmle / n1 + r2_cmle / n2
    stat = (rate1 - rate2 - value) / np.sqrt(v + eps)
    if return_cmle:
        return (stat, r1_cmle, r2_cmle)
    else:
        return stat