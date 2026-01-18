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
@pytest.mark.xfail(sys.platform == 'darwin' and blas_provider == 'openblas' and (blas_version < '0.3.21.dev'), reason='gges[float32] broken for OpenBLAS on macOS, see gh-16949')
def test_qz_single(self):
    rng = np.random.RandomState(12345)
    n = 5
    A = rng.random([n, n]).astype(float32)
    B = rng.random([n, n]).astype(float32)
    AA, BB, Q, Z = qz(A, B)
    assert_array_almost_equal(Q @ AA @ Z.T, A, decimal=5)
    assert_array_almost_equal(Q @ BB @ Z.T, B, decimal=5)
    assert_array_almost_equal(Q @ Q.T, eye(n), decimal=5)
    assert_array_almost_equal(Z @ Z.T, eye(n), decimal=5)
    assert_(np.all(diag(BB) >= 0))