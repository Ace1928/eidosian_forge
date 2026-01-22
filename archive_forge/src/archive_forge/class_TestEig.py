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
class TestEig:

    def test_simple(self):
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
        w, v = eig(a)
        exact_w = [(9 + sqrt(93)) / 2, 0, (9 - sqrt(93)) / 2]
        v0 = array([1, 1, (1 + sqrt(93) / 3) / 2])
        v1 = array([3.0, 0, -1])
        v2 = array([1, 1, (1 - sqrt(93) / 3) / 2])
        v0 = v0 / norm(v0)
        v1 = v1 / norm(v1)
        v2 = v2 / norm(v2)
        assert_array_almost_equal(w, exact_w)
        assert_array_almost_equal(v0, v[:, 0] * sign(v[0, 0]))
        assert_array_almost_equal(v1, v[:, 1] * sign(v[0, 1]))
        assert_array_almost_equal(v2, v[:, 2] * sign(v[0, 2]))
        for i in range(3):
            assert_array_almost_equal(a @ v[:, i], w[i] * v[:, i])
        w, v = eig(a, left=1, right=0)
        for i in range(3):
            assert_array_almost_equal(a.T @ v[:, i], w[i] * v[:, i])

    def test_simple_complex_eig(self):
        a = array([[1, 2], [-2, 1]])
        w, vl, vr = eig(a, left=1, right=1)
        assert_array_almost_equal(w, array([1 + 2j, 1 - 2j]))
        for i in range(2):
            assert_array_almost_equal(a @ vr[:, i], w[i] * vr[:, i])
        for i in range(2):
            assert_array_almost_equal(a.conj().T @ vl[:, i], w[i].conj() * vl[:, i])

    def test_simple_complex(self):
        a = array([[1, 2, 3], [1, 2, 3], [2, 5, 6 + 1j]])
        w, vl, vr = eig(a, left=1, right=1)
        for i in range(3):
            assert_array_almost_equal(a @ vr[:, i], w[i] * vr[:, i])
        for i in range(3):
            assert_array_almost_equal(a.conj().T @ vl[:, i], w[i].conj() * vl[:, i])

    def test_gh_3054(self):
        a = [[1]]
        b = [[0]]
        w, vr = eig(a, b, homogeneous_eigvals=True)
        assert_allclose(w[1, 0], 0)
        assert_(w[0, 0] != 0)
        assert_allclose(vr, 1)
        w, vr = eig(a, b)
        assert_equal(w, np.inf)
        assert_allclose(vr, 1)

    def _check_gen_eig(self, A, B):
        if B is not None:
            A, B = (asarray(A), asarray(B))
            B0 = B
        else:
            A = asarray(A)
            B0 = B
            B = np.eye(*A.shape)
        msg = f'\n{A!r}\n{B!r}'
        w, vr = eig(A, B0, homogeneous_eigvals=True)
        wt = eigvals(A, B0, homogeneous_eigvals=True)
        val1 = A @ vr * w[1, :]
        val2 = B @ vr * w[0, :]
        for i in range(val1.shape[1]):
            assert_allclose(val1[:, i], val2[:, i], rtol=1e-13, atol=1e-13, err_msg=msg)
        if B0 is None:
            assert_allclose(w[1, :], 1)
            assert_allclose(wt[1, :], 1)
        perm = np.lexsort(w)
        permt = np.lexsort(wt)
        assert_allclose(w[:, perm], wt[:, permt], atol=1e-07, rtol=1e-07, err_msg=msg)
        length = np.empty(len(vr))
        for i in range(len(vr)):
            length[i] = norm(vr[:, i])
        assert_allclose(length, np.ones(length.size), err_msg=msg, atol=1e-07, rtol=1e-07)
        beta_nonzero = w[1, :] != 0
        wh = w[0, beta_nonzero] / w[1, beta_nonzero]
        w, vr = eig(A, B0)
        wt = eigvals(A, B0)
        val1 = A @ vr
        val2 = B @ vr * w
        res = val1 - val2
        for i in range(res.shape[1]):
            if np.all(isfinite(res[:, i])):
                assert_allclose(res[:, i], 0, rtol=1e-13, atol=1e-13, err_msg=msg)
        w_fin = w[isfinite(w)]
        wt_fin = wt[isfinite(wt)]
        perm = argsort(clear_fuss(w_fin))
        permt = argsort(clear_fuss(wt_fin))
        assert_allclose(w[perm], wt[permt], atol=1e-07, rtol=1e-07, err_msg=msg)
        length = np.empty(len(vr))
        for i in range(len(vr)):
            length[i] = norm(vr[:, i])
        assert_allclose(length, np.ones(length.size), err_msg=msg)
        assert_allclose(sort(wh), sort(w[np.isfinite(w)]))

    @pytest.mark.xfail(reason='See gh-2254')
    def test_singular(self):
        A = array([[22, 34, 31, 31, 17], [45, 45, 42, 19, 29], [39, 47, 49, 26, 34], [27, 31, 26, 21, 15], [38, 44, 44, 24, 30]])
        B = array([[13, 26, 25, 17, 24], [31, 46, 40, 26, 37], [26, 40, 19, 25, 25], [16, 25, 27, 14, 23], [24, 35, 18, 21, 22]])
        with np.errstate(all='ignore'):
            self._check_gen_eig(A, B)

    def test_falker(self):
        M = diag(array([1, 0, 3]))
        K = array(([2, -1, -1], [-1, 2, -1], [-1, -1, 2]))
        D = array(([1, -1, 0], [-1, 1, 0], [0, 0, 0]))
        Z = zeros((3, 3))
        I3 = eye(3)
        A = np.block([[I3, Z], [Z, -K]])
        B = np.block([[Z, I3], [M, D]])
        with np.errstate(all='ignore'):
            self._check_gen_eig(A, B)

    def test_bad_geneig(self):

        def matrices(omega):
            c1 = -9 + omega ** 2
            c2 = 2 * omega
            A = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, c1, 0], [0, 0, 0, c1]]
            B = [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, -c2], [0, 1, c2, 0]]
            return (A, B)
        with np.errstate(all='ignore'):
            for k in range(100):
                A, B = matrices(omega=k * 5.0 / 100)
                self._check_gen_eig(A, B)

    def test_make_eigvals(self):
        rng = np.random.RandomState(1234)
        A = symrand(3, rng)
        self._check_gen_eig(A, None)
        B = symrand(3, rng)
        self._check_gen_eig(A, B)
        A = rng.random((3, 3)) + 1j * rng.random((3, 3))
        self._check_gen_eig(A, None)
        B = rng.random((3, 3)) + 1j * rng.random((3, 3))
        self._check_gen_eig(A, B)

    def test_check_finite(self):
        a = [[1, 2, 3], [1, 2, 3], [2, 5, 6]]
        w, v = eig(a, check_finite=False)
        exact_w = [(9 + sqrt(93)) / 2, 0, (9 - sqrt(93)) / 2]
        v0 = array([1, 1, (1 + sqrt(93) / 3) / 2])
        v1 = array([3.0, 0, -1])
        v2 = array([1, 1, (1 - sqrt(93) / 3) / 2])
        v0 = v0 / norm(v0)
        v1 = v1 / norm(v1)
        v2 = v2 / norm(v2)
        assert_array_almost_equal(w, exact_w)
        assert_array_almost_equal(v0, v[:, 0] * sign(v[0, 0]))
        assert_array_almost_equal(v1, v[:, 1] * sign(v[0, 1]))
        assert_array_almost_equal(v2, v[:, 2] * sign(v[0, 2]))
        for i in range(3):
            assert_array_almost_equal(a @ v[:, i], w[i] * v[:, i])

    def test_not_square_error(self):
        """Check that passing a non-square array raises a ValueError."""
        A = np.arange(6).reshape(3, 2)
        assert_raises(ValueError, eig, A)

    def test_shape_mismatch(self):
        """Check that passing arrays of with different shapes
        raises a ValueError."""
        A = eye(2)
        B = np.arange(9.0).reshape(3, 3)
        assert_raises(ValueError, eig, A, B)
        assert_raises(ValueError, eig, B, A)