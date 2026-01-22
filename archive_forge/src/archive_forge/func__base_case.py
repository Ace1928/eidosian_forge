import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def _base_case(m, n, k):
    """Base cases from recursive version"""
    if fmnks.get((m, n, k), -1) >= 0:
        return fmnks[m, n, k]
    elif k < 0 or m < 0 or n < 0 or (k > m * n):
        return 0
    elif k == 0 and m >= 0 and (n >= 0):
        return 1
    return None