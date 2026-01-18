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
def test_out_kwarg_w_cval_np_attr(self):
    """ Test issue #7286 where the cval is a np attr/string-based numerical
        constant"""
    for cval in (np.nan, np.inf, -np.inf, float('inf'), -float('inf')):

        def kernel(a):
            return a[0, 0] - a[1, 0]
        stencil_fn = numba.stencil(kernel, cval=cval)

        def wrapped():
            A = np.arange(12.0).reshape((3, 4))
            ret = np.ones_like(A)
            stencil_fn(A, out=ret)
            return ret
        A = np.arange(12.0).reshape((3, 4))
        expected = np.full_like(A, -4)
        expected[-1, :] = cval
        ret = np.ones_like(A)
        stencil_fn(A, out=ret)
        np.testing.assert_almost_equal(ret, expected)
        impls = self.compile_all(wrapped)
        for impl in impls:
            got = impl.entry_point()
            np.testing.assert_almost_equal(got, expected)