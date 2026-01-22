import functools
import itertools
import sys
import warnings
import threading
import operator
import numpy as np
import unittest
from numba import guvectorize, njit, typeof, vectorize
from numba.core import types
from numba.np.numpy_support import from_dtype
from numba.core.errors import LoweringError, TypingError
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.typing.npydecl import supported_ufuncs
from numba.np import numpy_support
from numba.core.registry import cpu_target
from numba.core.base import BaseContext
from numba.np import ufunc_db
class TestUfuncIssues(TestCase):

    def test_issue_651(self):

        @vectorize(['(float64,float64)'])
        def foo(x1, x2):
            return np.add(x1, x2) + np.add(x1, x2)
        a = np.arange(10, dtype='f8')
        b = np.arange(10, dtype='f8')
        self.assertPreciseEqual(foo(a, b), a + b + (a + b))

    def test_issue_2006(self):
        """
        <float32 ** int> should return float32, not float64.
        """

        def foo(x, y):
            return np.power(x, y)
        pyfunc = foo
        cfunc = njit(pyfunc)

        def check(x, y):
            got = cfunc(x, y)
            np.testing.assert_array_almost_equal(got, pyfunc(x, y))
            self.assertEqual(got.dtype, x.dtype)
        xs = [np.float32([1, 2, 3]), np.complex64([1j, 2, 3 - 3j])]
        for x in xs:
            check(x, 3)
            check(x, np.uint64(3))
            check(x, np.int64([2, 2, 3]))