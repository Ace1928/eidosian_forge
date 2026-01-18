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
def test_out_kwarg_w_cval(self):
    """ Issue #3518, out kwarg did not work with cval."""
    const_vals = [7, 7.0]

    def kernel(a):
        return a[0, 0] - a[1, 0]
    for const_val in const_vals:
        stencil_fn = numba.stencil(kernel, cval=const_val)

        def wrapped():
            A = np.arange(12).reshape((3, 4))
            ret = np.ones_like(A)
            stencil_fn(A, out=ret)
            return ret
        A = np.arange(12).reshape((3, 4))
        expected = np.full_like(A, -4)
        expected[-1, :] = const_val
        ret = np.ones_like(A)
        stencil_fn(A, out=ret)
        np.testing.assert_almost_equal(ret, expected)
        impls = self.compile_all(wrapped)
        for impl in impls:
            got = impl.entry_point()
            np.testing.assert_almost_equal(got, expected)
    stencil_fn = numba.stencil(kernel, cval=1j)

    def wrapped():
        A = np.arange(12).reshape((3, 4))
        ret = np.ones_like(A)
        stencil_fn(A, out=ret)
        return ret
    A = np.arange(12).reshape((3, 4))
    ret = np.ones_like(A)
    with self.assertRaises(NumbaValueError) as e:
        stencil_fn(A, out=ret)
    msg = 'cval type does not match stencil return type.'
    self.assertIn(msg, str(e.exception))
    for compiler in [self.compile_njit, self.compile_parallel]:
        try:
            compiler(wrapped, ())
        except (NumbaValueError, LoweringError) as e:
            self.assertIn(msg, str(e))
        else:
            raise AssertionError('Expected error was not raised')