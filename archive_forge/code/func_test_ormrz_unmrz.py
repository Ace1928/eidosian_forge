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
def test_ormrz_unmrz():
    """
    This test performs a matrix multiplication with an arbitrary m x n matrix C
    and a unitary matrix Q without explicitly forming the array. The array data
    is encoded in the rectangular part of A which is obtained from ?TZRZF. Q
    size is inferred by m, n, side keywords.
    """
    seed(1234)
    qm, qn, cn = (10, 15, 15)
    for ind, dtype in enumerate(DTYPES):
        tzrzf, tzrzf_lw = get_lapack_funcs(('tzrzf', 'tzrzf_lwork'), dtype=dtype)
        lwork_rz = _compute_lwork(tzrzf_lw, qm, qn)
        if ind < 2:
            A = triu(rand(qm, qn).astype(dtype))
            C = rand(cn, cn).astype(dtype)
            orun_mrz, orun_mrz_lw = get_lapack_funcs(('ormrz', 'ormrz_lwork'), dtype=dtype)
        else:
            A = triu((rand(qm, qn) + rand(qm, qn) * 1j).astype(dtype))
            C = (rand(cn, cn) + rand(cn, cn) * 1j).astype(dtype)
            orun_mrz, orun_mrz_lw = get_lapack_funcs(('unmrz', 'unmrz_lwork'), dtype=dtype)
        lwork_mrz = _compute_lwork(orun_mrz_lw, cn, cn)
        rz, tau, info = tzrzf(A, lwork=lwork_rz)
        V = np.hstack((np.eye(qm, dtype=dtype), rz[:, qm:]))
        Id = np.eye(qn, dtype=dtype)
        ref = [Id - tau[x] * V[[x], :].T.dot(V[[x], :].conj()) for x in range(qm)]
        Q = reduce(np.dot, ref)
        trans = 'T' if ind < 2 else 'C'
        tol = 10 * np.spacing(dtype(1.0).real)
        cq, info = orun_mrz(rz, tau, C, lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - Q.dot(C), zeros_like(C), atol=tol, rtol=0.0)
        cq, info = orun_mrz(rz, tau, C, trans=trans, lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - Q.conj().T.dot(C), zeros_like(C), atol=tol, rtol=0.0)
        cq, info = orun_mrz(rz, tau, C, side='R', lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - C.dot(Q), zeros_like(C), atol=tol, rtol=0.0)
        cq, info = orun_mrz(rz, tau, C, side='R', trans=trans, lwork=lwork_mrz)
        assert_(info == 0)
        assert_allclose(cq - C.dot(Q.conj().T), zeros_like(C), atol=tol, rtol=0.0)