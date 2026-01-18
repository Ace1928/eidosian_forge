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
@pytest.mark.parametrize('dtype,realtype', zip(DTYPES, REAL_DTYPES + REAL_DTYPES))
@pytest.mark.parametrize('compute_z', range(3))
def test_pteqr(dtype, realtype, compute_z):
    """
    Tests the ?pteqr lapack routine for all dtypes and compute_z parameters.
    It generates random SPD matrix diagonals d and e, and then confirms
    correct eigenvalues with scipy.linalg.eig. With applicable compute_z=2 it
    tests that z can reform A.
    """
    seed(42)
    atol = 1000 * np.finfo(dtype).eps
    pteqr = get_lapack_funcs('pteqr', dtype=dtype)
    n = 10
    d, e, A, z = pteqr_get_d_e_A_z(dtype, realtype, n, compute_z)
    d_pteqr, e_pteqr, z_pteqr, info = pteqr(d=d, e=e, z=z, compute_z=compute_z)
    assert_equal(info, 0, f'info = {info}, should be 0.')
    assert_allclose(np.sort(eigh(A)[0]), np.sort(d_pteqr), atol=atol)
    if compute_z:
        assert_allclose(z_pteqr @ np.conj(z_pteqr).T, np.identity(n), atol=atol)
        assert_allclose(z_pteqr @ np.diag(d_pteqr) @ np.conj(z_pteqr).T, A, atol=atol)