from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def array_sum_const_multi(arr, axis):
    a = np.sum(arr, axis=4)
    b = np.sum(arr, 3)
    c = np.sum(arr, axis)
    d = arr.sum(axis=5)
    e = np.sum(arr, axis=-1)
    return (a, b, c, d, e)