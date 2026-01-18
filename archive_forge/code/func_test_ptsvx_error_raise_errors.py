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
def test_ptsvx_error_raise_errors(dtype, realtype, fact, df_de_lambda):
    seed(42)
    ptsvx = get_lapack_funcs('ptsvx', dtype=dtype)
    n = 5
    d = generate_random_dtype_array((n,), realtype) + 4
    e = generate_random_dtype_array((n - 1,), dtype)
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    x_soln = generate_random_dtype_array((n, 2), dtype=dtype)
    b = A @ x_soln
    df, ef, info = df_de_lambda(d, e)
    assert_raises(ValueError, ptsvx, d[:-1], e, b, fact=fact, df=df, ef=ef)
    assert_raises(ValueError, ptsvx, d, e[:-1], b, fact=fact, df=df, ef=ef)
    assert_raises(Exception, ptsvx, d, e, b[:-1], fact=fact, df=df, ef=ef)