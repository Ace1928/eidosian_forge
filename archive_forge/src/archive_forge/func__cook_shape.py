import functools
import math
import warnings
import numpy as np
import cupy
from cupy.cuda import cufft
from cupy.fft import config
from cupy.fft._cache import get_plan_cache
def _cook_shape(a, s, axes, value_type, order='C'):
    if s is None or s == a.shape:
        return a
    if value_type == 'C2R' and s[-1] is not None:
        s = list(s)
        s[-1] = s[-1] // 2 + 1
    for sz, axis in zip(s, axes):
        if sz is not None and sz != a.shape[axis]:
            shape = list(a.shape)
            if shape[axis] > sz:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, sz)
                a = a[tuple(index)]
            else:
                index = [slice(None)] * a.ndim
                index[axis] = slice(0, shape[axis])
                shape[axis] = sz
                z = cupy.zeros(shape, a.dtype.char, order=order)
                z[tuple(index)] = a
                a = z
    return a