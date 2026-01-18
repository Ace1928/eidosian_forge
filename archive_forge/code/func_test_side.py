import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
@pytest.mark.parametrize('dtype_', DTYPES)
def test_side(self, dtype_):
    trmm = get_blas_funcs('trmm', dtype=dtype_)
    assert_raises(Exception, trmm, 1.0, self.a2, self.b2)
    res = trmm(1.0, self.a2.astype(dtype_), self.b2.astype(dtype_), side=1)
    k = self.b2.shape[1]
    assert_allclose(res, self.b2 @ self.a2[:k, :k], rtol=0.0, atol=100 * np.finfo(dtype_).eps)