import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
@skip_unsupported
def test_stencil_call(self):
    """Tests 2D numba.stencil calls.
        """

    def test_impl1(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.zeros(n ** 2).reshape((n, n))
        numba.stencil(lambda a: 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0]))(A, out=B)
        return B

    def test_impl2(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.zeros(n ** 2).reshape((n, n))

        def sf(a):
            return 0.25 * (a[0, 1] + a[1, 0] + a[0, -1] + a[-1, 0])
        B = numba.stencil(sf)(A)
        return B

    def test_impl_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.zeros(n ** 2).reshape((n, n))
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                B[i, j] = 0.25 * (A[i, j + 1] + A[i + 1, j] + A[i, j - 1] + A[i - 1, j])
        return B
    n = 100
    self.check(test_impl_seq, test_impl1, n)
    self.check(test_impl_seq, test_impl2, n)