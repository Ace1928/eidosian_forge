import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_unary_ufunc_1d_manual(self):

    def check(a, b):
        a_orig = a.copy()
        b_orig = b.copy()
        b0 = b.copy()
        c1 = ufunc(a, out=b0)
        c2 = ufunc(a, out=b)
        assert_array_equal(c1, c2)
        mask = view_element_first_byte(b).view(np.bool_)
        a[...] = a_orig
        b[...] = b_orig
        c1 = ufunc(a, out=b.copy(), where=mask.copy()).copy()
        a[...] = a_orig
        b[...] = b_orig
        c2 = ufunc(a, out=b, where=mask.copy()).copy()
        a[...] = a_orig
        b[...] = b_orig
        c3 = ufunc(a, out=b, where=mask).copy()
        assert_array_equal(c1, c2)
        assert_array_equal(c1, c3)
    dtypes = [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
    dtypes = [np.dtype(x) for x in dtypes]
    for dtype in dtypes:
        if np.issubdtype(dtype, np.integer):
            ufunc = np.invert
        else:
            ufunc = np.reciprocal
        n = 1000
        k = 10
        indices = [np.index_exp[:n], np.index_exp[k:k + n], np.index_exp[n - 1::-1], np.index_exp[k + n - 1:k - 1:-1], np.index_exp[:2 * n:2], np.index_exp[k:k + 2 * n:2], np.index_exp[2 * n - 1::-2], np.index_exp[k + 2 * n - 1:k - 1:-2]]
        for xi, yi in itertools.product(indices, indices):
            v = np.arange(1, 1 + n * 2 + k, dtype=dtype)
            x = v[xi]
            y = v[yi]
            with np.errstate(all='ignore'):
                check(x, y)
                check(x[:1], y)
                check(x[-1:], y)
                check(x[:1].reshape([]), y)
                check(x[-1:].reshape([]), y)