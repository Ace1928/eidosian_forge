import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
class TestHetrd:

    @pytest.mark.parametrize('complex_dtype', COMPLEX_DTYPES)
    def test_hetrd_with_zero_dim_array(self, complex_dtype):
        A = np.zeros((0, 0), dtype=complex_dtype)
        hetrd = get_lapack_funcs('hetrd', (A,))
        assert_raises(ValueError, hetrd, A)

    @pytest.mark.parametrize('real_dtype,complex_dtype', zip(REAL_DTYPES, COMPLEX_DTYPES))
    @pytest.mark.parametrize('n', (1, 3))
    def test_hetrd(self, n, real_dtype, complex_dtype):
        A = np.zeros((n, n), dtype=complex_dtype)
        hetrd, hetrd_lwork = get_lapack_funcs(('hetrd', 'hetrd_lwork'), (A,))
        A[np.triu_indices_from(A)] = np.arange(1, n * (n + 1) // 2 + 1, dtype=real_dtype) + 1j * np.arange(1, n * (n + 1) // 2 + 1, dtype=real_dtype)
        np.fill_diagonal(A, np.real(np.diag(A)))
        for x in [0, 1]:
            _, info = hetrd_lwork(n, lower=x)
            assert_equal(info, 0)
        lwork = _compute_lwork(hetrd_lwork, n)
        data, d, e, tau, info = hetrd(A, lower=1, lwork=lwork)
        assert_equal(info, 0)
        assert_allclose(data, A, atol=5 * np.finfo(real_dtype).eps, rtol=1.0)
        assert_allclose(d, np.real(np.diag(A)))
        assert_allclose(e, 0.0)
        assert_allclose(tau, 0.0)
        data, d, e, tau, info = hetrd(A, lwork=lwork)
        assert_equal(info, 0)
        T = np.zeros_like(A, dtype=real_dtype)
        k = np.arange(A.shape[0], dtype=int)
        T[k, k] = d
        k2 = np.arange(A.shape[0] - 1, dtype=int)
        T[k2 + 1, k2] = e
        T[k2, k2 + 1] = e
        Q = np.eye(n, n, dtype=complex_dtype)
        for i in range(n - 1):
            v = np.zeros(n, dtype=complex_dtype)
            v[:i] = data[:i, i + 1]
            v[i] = 1.0
            H = np.eye(n, n, dtype=complex_dtype) - tau[i] * np.outer(v, np.conj(v))
            Q = np.dot(H, Q)
        i_lower = np.tril_indices(n, -1)
        A[i_lower] = np.conj(A.T[i_lower])
        QHAQ = np.dot(np.conj(Q.T), np.dot(A, Q))
        assert_allclose(QHAQ, T, atol=10 * np.finfo(real_dtype).eps, rtol=1.0)