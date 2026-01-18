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
@pytest.mark.parametrize('fact,df_de_lambda', [('F', lambda d, e: get_lapack_funcs('pttrf', dtype=e.dtype)(d, e)), ('N', lambda d, e: (None, None, None))])
def test_ptsvx(dtype, realtype, fact, df_de_lambda):
    """
    This tests the ?ptsvx lapack routine wrapper to solve a random system
    Ax = b for all dtypes and input variations. Tests for: unmodified
    input parameters, fact options, incompatible matrix shapes raise an error,
    and singular matrices return info of illegal value.
    """
    seed(42)
    atol = 100 * np.finfo(dtype).eps
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)
    n = 5
    d = generate_random_dtype_array((n,), realtype) + 4
    e = generate_random_dtype_array((n - 1,), dtype)
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)
    b = A @ x_soln
    df, ef, info = df_de_lambda(d, e)
    diag_cpy = [d.copy(), e.copy(), b.copy()]
    df, ef, x, rcond, ferr, berr, info = ptsvx(d, e, b, fact=fact, df=df, ef=ef)
    assert_array_equal(d, diag_cpy[0])
    assert_array_equal(e, diag_cpy[1])
    assert_array_equal(b, diag_cpy[2])
    assert_(info == 0, f'info should be 0 but is {info}.')
    assert_array_almost_equal(x_soln, x)
    L = np.diag(ef, -1) + np.diag(np.ones(n))
    D = np.diag(df)
    assert_allclose(A, L @ D @ np.conj(L).T, atol=atol)
    assert not hasattr(rcond, '__len__'), f'rcond should be scalar but is {rcond}'
    assert_(ferr.shape == (2,), 'ferr.shape is {} but should be ({},)'.format(ferr.shape, x_soln.shape[1]))
    assert_(berr.shape == (2,), 'berr.shape is {} but should be ({},)'.format(berr.shape, x_soln.shape[1]))