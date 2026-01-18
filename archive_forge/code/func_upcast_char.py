import sys
import operator
import numpy as np
from math import prod
import scipy.sparse as sp
from scipy._lib._util import np_long, np_ulong
def upcast_char(*args):
    """Same as `upcast` but taking dtype.char as input (faster)."""
    t = _upcast_memo.get(args)
    if t is not None:
        return t
    t = upcast(*map(np.dtype, args))
    _upcast_memo[args] = t
    return t