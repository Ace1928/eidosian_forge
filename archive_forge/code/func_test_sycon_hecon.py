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
def test_sycon_hecon():
    seed(1234)
    for ind, dtype in enumerate(DTYPES + COMPLEX_DTYPES):
        n = 10
        if ind < 4:
            func_lwork = get_lapack_funcs('sytrf_lwork', dtype=dtype)
            funcon, functrf = get_lapack_funcs(('sycon', 'sytrf'), dtype=dtype)
            A = rand(n, n).astype(dtype)
        else:
            func_lwork = get_lapack_funcs('hetrf_lwork', dtype=dtype)
            funcon, functrf = get_lapack_funcs(('hecon', 'hetrf'), dtype=dtype)
            A = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
        A = (A + A.conj().T) / 2 + 2 * np.eye(n, dtype=dtype)
        anorm = norm(A, 1)
        lwork = _compute_lwork(func_lwork, n)
        ldu, ipiv, _ = functrf(A, lwork=lwork, lower=1)
        rcond, _ = funcon(a=ldu, ipiv=ipiv, anorm=anorm, lower=1)
        assert_(abs(1 / rcond - np.linalg.cond(A, p=1)) * rcond < 1)