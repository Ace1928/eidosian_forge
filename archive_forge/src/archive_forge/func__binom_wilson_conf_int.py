from math import sqrt
import numpy as np
from scipy._lib._util import _validate_int
from scipy.optimize import brentq
from scipy.special import ndtri
from ._discrete_distns import binom
from ._common import ConfidenceInterval
def _binom_wilson_conf_int(k, n, confidence_level, alternative, correction):
    p = k / n
    if alternative == 'two-sided':
        z = ndtri(0.5 + 0.5 * confidence_level)
    else:
        z = ndtri(confidence_level)
    denom = 2 * (n + z ** 2)
    center = (2 * n * p + z ** 2) / denom
    q = 1 - p
    if correction:
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            dlo = (1 + z * sqrt(z ** 2 - 2 - 1 / n + 4 * p * (n * q + 1))) / denom
            lo = center - dlo
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            dhi = (1 + z * sqrt(z ** 2 + 2 - 1 / n + 4 * p * (n * q - 1))) / denom
            hi = center + dhi
    else:
        delta = z / denom * sqrt(4 * n * p * q + z ** 2)
        if alternative == 'less' or k == 0:
            lo = 0.0
        else:
            lo = center - delta
        if alternative == 'greater' or k == n:
            hi = 1.0
        else:
            hi = center + delta
    return (lo, hi)