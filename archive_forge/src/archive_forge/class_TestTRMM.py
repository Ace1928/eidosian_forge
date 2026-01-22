import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
class TestTRMM:
    """Quick and simple tests for dtrmm."""

    def setup_method(self):
        self.a = np.array([[1.0, 2.0], [-2.0, 1.0]])
        self.b = np.array([[3.0, 4.0, -1.0], [5.0, 6.0, -2.0]])
        self.a2 = np.array([[1, 1, 2, 3], [0, 1, 4, 5], [0, 0, 1, 6], [0, 0, 0, 1]], order='f')
        self.b2 = np.array([[1, 4], [2, 5], [3, 6], [7, 8], [9, 10]], order='f')

    @pytest.mark.parametrize('dtype_', DTYPES)
    def test_side(self, dtype_):
        trmm = get_blas_funcs('trmm', dtype=dtype_)
        assert_raises(Exception, trmm, 1.0, self.a2, self.b2)
        res = trmm(1.0, self.a2.astype(dtype_), self.b2.astype(dtype_), side=1)
        k = self.b2.shape[1]
        assert_allclose(res, self.b2 @ self.a2[:k, :k], rtol=0.0, atol=100 * np.finfo(dtype_).eps)

    def test_ab(self):
        f = getattr(fblas, 'dtrmm', None)
        if f is not None:
            result = f(1.0, self.a, self.b)
            expected = np.array([[13.0, 16.0, -5.0], [5.0, 6.0, -2.0]])
            assert_array_almost_equal(result, expected)

    def test_ab_lower(self):
        f = getattr(fblas, 'dtrmm', None)
        if f is not None:
            result = f(1.0, self.a, self.b, lower=True)
            expected = np.array([[3.0, 4.0, -1.0], [-1.0, -2.0, 0.0]])
            assert_array_almost_equal(result, expected)

    def test_b_overwrites(self):
        f = getattr(fblas, 'dtrmm', None)
        if f is not None:
            for overwr in [True, False]:
                bcopy = self.b.copy()
                result = f(1.0, self.a, bcopy, overwrite_b=overwr)
                assert_(bcopy.flags.f_contiguous is False and np.may_share_memory(bcopy, result) is False)
                assert_equal(bcopy, self.b)
            bcopy = np.asfortranarray(self.b.copy())
            result = f(1.0, self.a, bcopy, overwrite_b=True)
            assert_(bcopy.flags.f_contiguous is True and np.may_share_memory(bcopy, result) is True)
            assert_array_almost_equal(bcopy, result)