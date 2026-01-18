import os
import re
import copy
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import pytest
from scipy.linalg import svd, null_space
from scipy.sparse import csc_matrix, issparse, spdiags, random
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg import svds
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence
@pytest.mark.filterwarnings('ignore:The problem size')
@pytest.mark.parametrize('dtype', (float, complex, np.float32))
def test_small_sigma2(self, dtype):
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not enabled')
        elif dtype == np.float32:
            pytest.skip('Test failures in CI, see gh-17004')
        elif dtype == complex:
            pytest.skip('PROPACK unsupported for complex dtype')
    rng = np.random.default_rng(179847540)
    dim = 4
    size = 10
    x = rng.random((size, size - dim))
    y = x[:, :dim] * rng.random(dim)
    mat = np.hstack((x, y))
    mat = mat.astype(dtype)
    nz = null_space(mat)
    assert_equal(nz.shape[1], dim)
    u, s, vh = svd(mat)
    assert_allclose(s[-dim:], 0, atol=1e-06, rtol=1.0)
    assert_allclose(mat @ vh[-dim:, :].T, 0, atol=1e-06, rtol=1.0)
    sp_mat = csc_matrix(mat)
    su, ss, svh = svds(sp_mat, k=dim, which='SM', solver=self.solver)
    assert_allclose(ss, 0, atol=1e-05, rtol=1.0)
    n, m = mat.shape
    if n < m:
        assert_allclose(sp_mat.transpose() @ su, 0, atol=1e-05, rtol=1.0)
    assert_allclose(sp_mat @ svh.T, 0, atol=1e-05, rtol=1.0)