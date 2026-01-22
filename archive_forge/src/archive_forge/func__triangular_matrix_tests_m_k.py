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
def _triangular_matrix_tests_m_k(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)

    def _check(arr):
        for k in itertools.chain.from_iterable(([None], range(-10, 10))):
            if k is None:
                params = {}
            else:
                params = {'k': k}
            expected = pyfunc(arr, **params)
            got = cfunc(arr, **params)
            self.assertEqual(got.dtype, expected.dtype)
            np.testing.assert_array_equal(got, expected)
    return self._triangular_matrix_tests_inner(self, pyfunc, _check)