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
def _nonneg_int_or_fail(n, var_name, strict=True):
    try:
        if strict:
            n = operator.index(n)
        elif n == floor(n):
            n = int(n)
        else:
            raise ValueError()
        if n < 0:
            raise ValueError()
    except (ValueError, TypeError) as err:
        raise err.__class__(f'{var_name} must be a non-negative integer') from err
    return n