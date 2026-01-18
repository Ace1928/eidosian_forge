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
def test_sfrk_hfrk():
    """
    Test for performing a symmetric rank-k operation for matrix in RFP format.
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
        prefix = 's' if ind < 2 else 'h'
        trttf, tfttr, shfrk = get_lapack_funcs(('trttf', 'tfttr', f'{prefix}frk'), dtype=dtype)
        Afp, _ = trttf(A)
        C = np.random.rand(n, 2).astype(dtype)
        Afp_out = shfrk(n, 2, -1, C, 2, Afp)
        A_out, _ = tfttr(n, Afp_out)
        assert_array_almost_equal(A_out, triu(-C.dot(C.conj().T) + 2 * A), decimal=4 if ind % 2 == 0 else 6)