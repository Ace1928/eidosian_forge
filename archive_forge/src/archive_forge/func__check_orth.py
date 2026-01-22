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
def _check_orth(n, dtype, skip_big=False):
    X = np.ones((n, 2), dtype=float).astype(dtype)
    eps = np.finfo(dtype).eps
    tol = 1000 * eps
    Y = orth(X)
    assert_equal(Y.shape, (n, 1))
    assert_allclose(Y, Y.mean(), atol=tol)
    Y = orth(X.T)
    assert_equal(Y.shape, (2, 1))
    assert_allclose(Y, Y.mean(), atol=tol)
    if n > 5 and (not skip_big):
        np.random.seed(1)
        X = np.random.rand(n, 5) @ np.random.rand(5, n)
        X = X + 0.0001 * np.random.rand(n, 1) @ np.random.rand(1, n)
        X = X.astype(dtype)
        Y = orth(X, rcond=0.001)
        assert_equal(Y.shape, (n, 5))
        Y = orth(X, rcond=1e-06)
        assert_equal(Y.shape, (n, 5 + 1))