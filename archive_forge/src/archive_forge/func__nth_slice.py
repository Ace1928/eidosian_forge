import operator
import functools
import warnings
import numpy as np
from numpy.core.multiarray import dragon4_positional, dragon4_scientific
from numpy.core.umath import absolute
def _nth_slice(i, ndim):
    sl = [np.newaxis] * ndim
    sl[i] = slice(None)
    return tuple(sl)