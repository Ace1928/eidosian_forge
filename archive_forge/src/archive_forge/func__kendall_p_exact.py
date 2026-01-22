import numpy as np
from numpy import ndarray
import numpy.ma as ma
from numpy.ma import masked, nomask
import math
import itertools
import warnings
from collections import namedtuple
from . import distributions
from scipy._lib._util import _rename_parameter, _contains_nan
from scipy._lib._bunch import _make_tuple_bunch
import scipy.special as special
import scipy.stats._stats_py
from ._stats_mstats_common import (
def _kendall_p_exact(n, c, alternative='two-sided'):
    in_right_tail = c >= n * (n - 1) // 2 - c
    alternative_greater = alternative == 'greater'
    c = int(min(c, n * (n - 1) // 2 - c))
    if n <= 0:
        raise ValueError(f'n ({n}) must be positive')
    elif c < 0 or 4 * c > n * (n - 1):
        raise ValueError(f'c ({c}) must satisfy 0 <= 4c <= n(n-1) = {n * (n - 1)}.')
    elif n == 1:
        prob = 1.0
        p_mass_at_c = 1
    elif n == 2:
        prob = 1.0
        p_mass_at_c = 0.5
    elif c == 0:
        prob = 2.0 / math.factorial(n) if n < 171 else 0.0
        p_mass_at_c = prob / 2
    elif c == 1:
        prob = 2.0 / math.factorial(n - 1) if n < 172 else 0.0
        p_mass_at_c = (n - 1) / math.factorial(n)
    elif 4 * c == n * (n - 1) and alternative == 'two-sided':
        prob = 1.0
    elif n < 171:
        new = np.zeros(c + 1)
        new[0:2] = 1.0
        for j in range(3, n + 1):
            new = np.cumsum(new)
            if j <= c:
                new[j:] -= new[:c + 1 - j]
        prob = 2.0 * np.sum(new) / math.factorial(n)
        p_mass_at_c = new[-1] / math.factorial(n)
    else:
        new = np.zeros(c + 1)
        new[0:2] = 1.0
        for j in range(3, n + 1):
            new = np.cumsum(new) / j
            if j <= c:
                new[j:] -= new[:c + 1 - j]
        prob = np.sum(new)
        p_mass_at_c = new[-1] / 2
    if alternative != 'two-sided':
        if in_right_tail == alternative_greater:
            prob /= 2
        else:
            prob = 1 - prob / 2 + p_mass_at_c
    prob = np.clip(prob, 0, 1)
    return prob