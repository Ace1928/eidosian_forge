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
def test_take_along_axis_broadcasting(self):
    arr = np.ones((3, 4, 1))
    ai = np.ones((1, 2, 5), dtype=np.intp)

    def gen(axis):

        @njit
        def impl(a, i):
            return np.take_along_axis(a, i, axis)
        return impl
    for i in (1, -2):
        check = gen(i)
        expected = check.py_func(arr, ai)
        actual = check(arr, ai)
        self.assertPreciseEqual(expected, actual)
        self.assertEqual(actual.shape, (3, 2, 5))