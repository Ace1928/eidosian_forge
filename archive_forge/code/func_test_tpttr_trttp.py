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
def test_tpttr_trttp():
    """
    Test conversion routines between the Rectengular Full Packed (RFP) format
    and Standard Triangular Array (TR)
    """
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 20
        if ind > 1:
            A_full = (rand(n, n) + rand(n, n) * 1j).astype(dtype)
        else:
            A_full = rand(n, n).astype(dtype)
        trttp, tpttr = get_lapack_funcs(('trttp', 'tpttr'), dtype=dtype)
        A_tp_U, info = trttp(A_full)
        assert_(info == 0)
        A_tp_L, info = trttp(A_full, uplo='L')
        assert_(info == 0)
        inds = tril_indices(n)
        A_tp_U_m = zeros(n * (n + 1) // 2, dtype=dtype)
        A_tp_U_m[:] = triu(A_full).T[inds]
        inds = triu_indices(n)
        A_tp_L_m = zeros(n * (n + 1) // 2, dtype=dtype)
        A_tp_L_m[:] = tril(A_full).T[inds]
        assert_array_almost_equal(A_tp_U, A_tp_U_m)
        assert_array_almost_equal(A_tp_L, A_tp_L_m)
        A_tr_U, info = tpttr(n, A_tp_U)
        assert_(info == 0)
        A_tr_L, info = tpttr(n, A_tp_L, uplo='L')
        assert_(info == 0)
        assert_array_almost_equal(A_tr_U, triu(A_full))
        assert_array_almost_equal(A_tr_L, tril(A_full))