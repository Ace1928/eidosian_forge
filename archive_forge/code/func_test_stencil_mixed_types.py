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
def test_stencil_mixed_types(self):

    def test_impl_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = n ** 2 - np.arange(n ** 2).reshape((n, n))
        S = np.eye(n, dtype=np.bool_)
        O = np.zeros((n, n), dtype=A.dtype)
        for i in range(0, n):
            for j in range(0, n):
                O[i, j] = A[i, j] if S[i, j] else B[i, j]
        return O

    def test_seq(n):
        A = np.arange(n ** 2).reshape((n, n))
        B = n ** 2 - np.arange(n ** 2).reshape((n, n))
        S = np.eye(n, dtype=np.bool_)
        O = stencil_multiple_input_mixed_types_2d(A, B, S)
        return O
    n = 3
    self.check(test_impl_seq, test_seq, n)