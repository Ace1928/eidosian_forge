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
def test_stencil_standard_indexing_1d(self):
    """Tests standard indexing with a 1d array.
        """

    def test_seq(n):
        A = np.arange(n)
        B = [3.0, 7.0]
        C = stencil_with_standard_indexing_1d(A, B)
        return C

    def test_impl_seq(n):
        A = np.arange(n)
        B = [3.0, 7.0]
        C = np.zeros(n)
        for i in range(1, n):
            C[i] = A[i - 1] * B[0] + A[i] * B[1]
        return C
    n = 100
    self.check(test_impl_seq, test_seq, n)