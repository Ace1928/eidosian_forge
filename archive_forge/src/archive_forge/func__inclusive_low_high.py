from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from scipy.special import expit, logit
from scipy.stats import gmean
from ..utils.extmath import softmax
def _inclusive_low_high(interval, dtype=np.float64):
    """Generate values low and high to be within the interval range.

    This is used in tests only.

    Returns
    -------
    low, high : tuple
        The returned values low and high lie within the interval.
    """
    eps = 10 * np.finfo(dtype).eps
    if interval.low == -np.inf:
        low = -10000000000.0
    elif interval.low < 0:
        low = interval.low * (1 - eps) + eps
    else:
        low = interval.low * (1 + eps) + eps
    if interval.high == np.inf:
        high = 10000000000.0
    elif interval.high < 0:
        high = interval.high * (1 + eps) - eps
    else:
        high = interval.high * (1 - eps) - eps
    return (low, high)