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
class TestSVD_GESDD:
    lapack_driver = 'gesdd'

    def test_degenerate(self):
        assert_raises(TypeError, svd, [[1.0]], lapack_driver=1.0)
        assert_raises(ValueError, svd, [[1.0]], lapack_driver='foo')

    def test_simple(self):
        a = [[1, 2, 3], [1, 20, 3], [2, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(3))
            assert_array_almost_equal(vh.T @ vh, eye(3))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_singular(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(3))
            assert_array_almost_equal(vh.T @ vh, eye(3))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_underdet(self):
        a = [[1, 2, 3], [4, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(u.shape[0]))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_overdet(self):
        a = [[1, 2], [4, 5], [3, 4]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.T @ u, eye(u.shape[1]))
            assert_array_almost_equal(vh.T @ vh, eye(2))
            sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_random(self):
        rng = np.random.RandomState(1234)
        n = 20
        m = 15
        for i in range(3):
            for a in [rng.random([n, m]), rng.random([m, n])]:
                for full_matrices in (True, False):
                    u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
                    assert_array_almost_equal(u.T @ u, eye(u.shape[1]))
                    assert_array_almost_equal(vh @ vh.T, eye(vh.shape[0]))
                    sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
                    for i in range(len(s)):
                        sigma[i, i] = s[i]
                    assert_array_almost_equal(u @ sigma @ vh, a)

    def test_simple_complex(self):
        a = [[1, 2, 3], [1, 2j, 3], [2, 5, 6]]
        for full_matrices in (True, False):
            u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
            assert_array_almost_equal(u.conj().T @ u, eye(u.shape[1]))
            assert_array_almost_equal(vh.conj().T @ vh, eye(vh.shape[0]))
            sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
            for i in range(len(s)):
                sigma[i, i] = s[i]
            assert_array_almost_equal(u @ sigma @ vh, a)

    def test_random_complex(self):
        rng = np.random.RandomState(1234)
        n = 20
        m = 15
        for i in range(3):
            for full_matrices in (True, False):
                for a in [rng.random([n, m]), rng.random([m, n])]:
                    a = a + 1j * rng.random(list(a.shape))
                    u, s, vh = svd(a, full_matrices=full_matrices, lapack_driver=self.lapack_driver)
                    assert_array_almost_equal(u.conj().T @ u, eye(u.shape[1]))
                    sigma = zeros((u.shape[1], vh.shape[0]), s.dtype.char)
                    for i in range(len(s)):
                        sigma[i, i] = s[i]
                    assert_array_almost_equal(u @ sigma @ vh, a)

    def test_crash_1580(self):
        rng = np.random.RandomState(1234)
        sizes = [(13, 23), (30, 50), (60, 100)]
        for sz in sizes:
            for dt in [np.float32, np.float64, np.complex64, np.complex128]:
                a = rng.rand(*sz).astype(dt)
                svd(a, lapack_driver=self.lapack_driver)

    def test_check_finite(self):
        a = [[1, 2, 3], [1, 20, 3], [2, 5, 6]]
        u, s, vh = svd(a, check_finite=False, lapack_driver=self.lapack_driver)
        assert_array_almost_equal(u.T @ u, eye(3))
        assert_array_almost_equal(vh.T @ vh, eye(3))
        sigma = zeros((u.shape[0], vh.shape[0]), s.dtype.char)
        for i in range(len(s)):
            sigma[i, i] = s[i]
        assert_array_almost_equal(u @ sigma @ vh, a)

    def test_gh_5039(self):
        b = np.array([[0.16666667, 0.66666667, 0.16666667, 0.0, 0.0, 0.0], [0.0, 0.16666667, 0.66666667, 0.16666667, 0.0, 0.0], [0.0, 0.0, 0.16666667, 0.66666667, 0.16666667, 0.0], [0.0, 0.0, 0.0, 0.16666667, 0.66666667, 0.16666667]])
        svd(b, lapack_driver=self.lapack_driver)

    @pytest.mark.skipif(not HAS_ILP64, reason='64-bit LAPACK required')
    @pytest.mark.slow
    def test_large_matrix(self):
        check_free_memory(free_mb=17000)
        A = np.zeros([1, 2 ** 31], dtype=np.float32)
        A[0, -1] = 1
        u, s, vh = svd(A, full_matrices=False)
        assert_allclose(s[0], 1.0)
        assert_allclose(u[0, 0] * vh[0, -1], 1.0)