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
def test_qz_double_sort(self):
    A = np.array([[3.9, 12.5, -34.5, 2.5], [4.3, 21.5, -47.5, 7.5], [4.3, 1.5, -43.5, 3.5], [4.4, 6.0, -46.0, 6.0]])
    B = np.array([[1.0, 1.0, -3.0, 1.0], [1.0, 3.0, -5.0, 4.4], [1.0, 2.0, -4.0, 1.0], [1.2, 3.0, -4.0, 4.0]])
    assert_raises(ValueError, qz, A, B, sort=lambda ar, ai, beta: ai == 0)
    if False:
        AA, BB, Q, Z, sdim = qz(A, B, sort=lambda ar, ai, beta: ai == 0)
        assert_(sdim == 4)
        assert_array_almost_equal(Q @ AA @ Z.T, A)
        assert_array_almost_equal(Q @ BB @ Z.T, B)
        assert_array_almost_equal(np.abs(AA), np.abs(np.array([[35.7864, -80.9061, -12.0629, -9.498], [0.0, 2.7638, -2.3505, 7.3256], [0.0, 0.0, 0.6258, -0.0398], [0.0, 0.0, 0.0, -12.8217]])), 4)
        assert_array_almost_equal(np.abs(BB), np.abs(np.array([[4.5324, -8.7878, 3.2357, -3.5526], [0.0, 1.4314, -2.1894, 0.9709], [0.0, 0.0, 1.3126, -0.3468], [0.0, 0.0, 0.0, 0.559]])), 4)
        assert_array_almost_equal(np.abs(Q), np.abs(np.array([[-0.4193, -0.605, -0.1894, -0.6498], [-0.5495, 0.6987, 0.2654, -0.3734], [-0.4973, -0.3682, 0.6194, 0.4832], [-0.5243, 0.1008, -0.7142, 0.4526]])), 4)
        assert_array_almost_equal(np.abs(Z), np.abs(np.array([[-0.9471, -0.2971, -0.1217, 0.0055], [-0.0367, 0.1209, 0.0358, 0.9913], [0.3171, -0.9041, -0.2547, 0.1312], [0.0346, 0.2824, -0.9587, 0.0014]])), 4)