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
def test_syconv():
    """
    Test for going back and forth between the returned format of he/sytrf to
    L and D factors/permutations.
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 10
        if ind > 1:
            A = (randint(-30, 30, (n, n)) + randint(-30, 30, (n, n)) * 1j).astype(dtype)
            A = A + A.conj().T
        else:
            A = randint(-30, 30, (n, n)).astype(dtype)
            A = A + A.T + n * eye(n)
        tol = 100 * np.spacing(dtype(1.0).real)
        syconv, trf, trf_lwork = get_lapack_funcs(('syconv', 'sytrf', 'sytrf_lwork'), dtype=dtype)
        lw = _compute_lwork(trf_lwork, n, lower=1)
        L, D, perm = ldl(A, lower=1, hermitian=False)
        lw = _compute_lwork(trf_lwork, n, lower=1)
        ldu, ipiv, info = trf(A, lower=1, lwork=lw)
        a, e, info = syconv(ldu, ipiv, lower=1)
        assert_allclose(tril(a, -1), tril(L[perm, :], -1), atol=tol, rtol=0.0)
        U, D, perm = ldl(A, lower=0, hermitian=False)
        ldu, ipiv, info = trf(A, lower=0)
        a, e, info = syconv(ldu, ipiv, lower=0)
        assert_allclose(triu(a, 1), triu(U[perm, :], 1), atol=tol, rtol=0.0)