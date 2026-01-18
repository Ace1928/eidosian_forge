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
def test_pftrf():
    """
    Test Cholesky factorization of a positive definite Rectengular Full
    Packed (RFP) format array
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
            A = A + A.conj().T + n * eye(n)
        else:
            A = rand(n, n).astype(dtype)
            A = A + A.T + n * eye(n)
        pftrf, trttf, tfttr = get_lapack_funcs(('pftrf', 'trttf', 'tfttr'), dtype=dtype)
        Afp, info = trttf(A)
        Achol_rfp, info = pftrf(n, Afp)
        assert_(info == 0)
        A_chol_r, _ = tfttr(n, Achol_rfp)
        Achol = cholesky(A)
        assert_array_almost_equal(A_chol_r, Achol)