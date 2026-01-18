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
def test_sing_val_update(self):
    sigmas = np.array([4.0, 3.0, 2.0, 0])
    m_vec = np.array([3.12, 5.7, -4.8, -2.2])
    M = np.hstack((np.vstack((np.diag(sigmas[0:-1]), np.zeros((1, len(m_vec) - 1)))), m_vec[:, np.newaxis]))
    SM = svd(M, full_matrices=False, compute_uv=False, overwrite_a=False, check_finite=False)
    it_len = len(sigmas)
    sgm = np.concatenate((sigmas[::-1], [sigmas[0] + it_len * norm(m_vec)]))
    mvc = np.concatenate((m_vec[::-1], (0,)))
    lasd4 = get_lapack_funcs('lasd4', (sigmas,))
    roots = []
    for i in range(0, it_len):
        res = lasd4(i, sgm, mvc)
        roots.append(res[1])
        assert_(res[3] <= 0, 'LAPACK root finding dlasd4 failed to find                                     the singular value %i' % i)
    roots = np.array(roots)[::-1]
    assert_((not np.any(np.isnan(roots)), 'There are NaN roots'))
    assert_allclose(SM, roots, atol=100 * np.finfo(np.float64).eps, rtol=100 * np.finfo(np.float64).eps)