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
def test_stencil_nested1(self):
    """Tests whether nested stencil decorator works.
        """

    @njit(parallel=True)
    def test_impl(n):

        @stencil
        def fun(a):
            c = 2
            return a[-c + 1]
        B = fun(n)
        return B

    def test_impl_seq(n):
        B = np.zeros(len(n), dtype=int)
        for i in range(1, len(n)):
            B[i] = n[i - 1]
        return B
    n = np.arange(10)
    np.testing.assert_equal(test_impl(n), test_impl_seq(n))