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
def test_stencil_standard_indexing_2d(self):
    """Tests standard indexing with a 2d array and multiple stencil calls.
        """

    def test_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.ones((3, 3))
        C = stencil_with_standard_indexing_2d(A, B)
        D = stencil_with_standard_indexing_2d(C, B)
        return D

    def test_impl_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = np.ones((3, 3))
        C = np.zeros(n ** 2).reshape((n, n))
        D = np.zeros(n ** 2).reshape((n, n))
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                C[i, j] = A[i, j + 1] * B[0, 1] + A[i + 1, j] * B[1, 0] + A[i, j - 1] * B[0, -1] + A[i - 1, j] * B[-1, 0]
        for i in range(1, n - 1):
            for j in range(1, n - 1):
                D[i, j] = C[i, j + 1] * B[0, 1] + C[i + 1, j] * B[1, 0] + C[i, j - 1] * B[0, -1] + C[i - 1, j] * B[-1, 0]
        return D
    n = 5
    self.check(test_impl_seq, test_seq, n)