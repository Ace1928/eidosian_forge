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
def _bessel_diff_formula(v, z, n, L, phase):
    v = asarray(v)
    p = 1.0
    s = L(v - n, z)
    for i in range(1, n + 1):
        p = phase * (p * (n - i + 1)) / i
        s += p * L(v - n + i * 2, z)
    return s / 2.0 ** n