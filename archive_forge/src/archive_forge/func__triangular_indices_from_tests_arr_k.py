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
def _triangular_indices_from_tests_arr_k(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)
    for dtype in [int, float, bool]:
        for n, m in itertools.product(range(10), range(10)):
            arr = np.ones((n, m), dtype)
            for k in range(-10, 10):
                expected = pyfunc(arr)
                got = cfunc(arr)
                self.assertEqual(type(expected), type(got))
                self.assertEqual(len(expected), len(got))
                for e, g in zip(expected, got):
                    np.testing.assert_array_equal(e, g)