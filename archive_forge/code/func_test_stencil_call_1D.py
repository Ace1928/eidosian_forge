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
def test_stencil_call_1D(self):
    """Tests 1D numba.stencil calls.
        """

    def test_impl(n):
        A = np.arange(n)
        B = np.zeros(n)
        numba.stencil(lambda a: 0.3 * (a[-1] + a[0] + a[1]))(A, out=B)
        return B

    def test_impl_seq(n):
        A = np.arange(n)
        B = np.zeros(n)
        for i in range(1, n - 1):
            B[i] = 0.3 * (A[i - 1] + A[i] + A[i + 1])
        return B
    n = 100
    self.check(test_impl_seq, test_impl, n)