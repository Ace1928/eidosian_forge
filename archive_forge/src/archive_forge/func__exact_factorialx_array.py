import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def _exact_factorialx_array(n, k=1):
    """
    Exact computation of factorial for an array.

    The factorials are computed in incremental fashion, by taking
    the sorted unique values of n and multiplying the intervening
    numbers between the different unique values.

    In other words, the factorial for the largest input is only
    computed once, with each other result computed in the process.

    k > 1 corresponds to the multifactorial.
    """
    un = np.unique(n)
    un = un[~np.isnan(un)]
    if np.isnan(n).any():
        dt = float
    elif k in _FACTORIALK_LIMITS_64BITS.keys():
        if un[-1] > _FACTORIALK_LIMITS_64BITS[k]:
            dt = object
        elif un[-1] > _FACTORIALK_LIMITS_32BITS[k]:
            dt = np.int64
        else:
            dt = np.dtype('long')
    else:
        dt = object
    out = np.empty_like(n, dtype=dt)
    un = un[un > 1]
    out[n < 2] = 1
    out[n < 0] = 0
    for lane in range(0, k):
        ul = un[un % k == lane] if k > 1 else un
        if ul.size:
            val = _range_prod(1, int(ul[0]), k=k)
            out[n == ul[0]] = val
            for i in range(len(ul) - 1):
                prev = ul[i]
                current = ul[i + 1]
                val *= _range_prod(int(prev + 1), int(current), k=k)
                out[n == current] = val
    if np.isnan(n).any():
        out = out.astype(np.float64)
        out[np.isnan(n)] = np.nan
    return out