import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def test_null_space():
    np.random.seed(1)
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    sizes = [1, 2, 3, 10, 100]
    for dt, n in itertools.product(dtypes, sizes):
        X = np.ones((2, n), dtype=dt)
        eps = np.finfo(dt).eps
        tol = 1000 * eps
        Y = null_space(X)
        assert_equal(Y.shape, (n, n - 1))
        assert_allclose(X @ Y, 0, atol=tol)
        Y = null_space(X.T)
        assert_equal(Y.shape, (2, 1))
        assert_allclose(X.T @ Y, 0, atol=tol)
        X = np.random.randn(1 + n // 2, n)
        Y = null_space(X)
        assert_equal(Y.shape, (n, n - 1 - n // 2))
        assert_allclose(X @ Y, 0, atol=tol)
        if n > 5:
            np.random.seed(1)
            X = np.random.rand(n, 5) @ np.random.rand(5, n)
            X = X + 0.0001 * np.random.rand(n, 1) @ np.random.rand(1, n)
            X = X.astype(dt)
            Y = null_space(X, rcond=0.001)
            assert_equal(Y.shape, (n, n - 5))
            Y = null_space(X, rcond=1e-06)
            assert_equal(Y.shape, (n, n - 6))