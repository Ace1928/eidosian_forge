import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_np_append(self):

    def arrays():
        yield (2, 2, None)
        yield (np.arange(10), 3, None)
        yield (np.arange(10), np.arange(3), None)
        yield (np.arange(10).reshape(5, 2), np.arange(3), None)
        yield (np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9]]), 0)
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        yield (arr, arr, 1)
    pyfunc = append
    cfunc = jit(nopython=True)(pyfunc)
    for arr, obj, axis in arrays():
        expected = pyfunc(arr, obj, axis)
        got = cfunc(arr, obj, axis)
        self.assertPreciseEqual(expected, got)