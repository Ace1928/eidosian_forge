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
def test_pstrf():
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 10
        r = 2
        pstrf = get_lapack_funcs('pstrf', dtype=dtype)
        if ind > 1:
            A = rand(n, n - r).astype(dtype) + 1j * rand(n, n - r).astype(dtype)
            A = A @ A.conj().T
        else:
            A = rand(n, n - r).astype(dtype)
            A = A @ A.T
        c, piv, r_c, info = pstrf(A)
        U = triu(c)
        U[r_c - n:, r_c - n:] = 0.0
        assert_equal(info, 1)
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        assert_allclose(A[piv - 1][:, piv - 1], U.conj().T @ U, rtol=0.0, atol=atol)
        c, piv, r_c, info = pstrf(A, lower=1)
        L = tril(c)
        L[r_c - n:, r_c - n:] = 0.0
        assert_equal(info, 1)
        single_atol = 1000 * np.finfo(np.float32).eps
        double_atol = 1000 * np.finfo(np.float64).eps
        atol = single_atol if ind in [0, 2] else double_atol
        assert_allclose(A[piv - 1][:, piv - 1], L @ L.conj().T, rtol=0.0, atol=atol)