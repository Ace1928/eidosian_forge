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
@pytest.mark.filterwarnings('ignore:Exited at iteration')
@pytest.mark.filterwarnings('ignore:Exited postprocessing')
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dtype', (float, complex, np.float32))
def test_small_sigma_sparse(self, shape, dtype):
    solver = self.solver
    if solver == 'propack':
        pytest.skip('PROPACK failures unrelated to PR')
    rng = np.random.default_rng(0)
    k = 5
    m, n = shape
    S = random(m, n, density=0.1, random_state=rng)
    if dtype == complex:
        S = +1j * random(m, n, density=0.1, random_state=rng)
    e = np.ones(m)
    e[0:5] *= 10.0 ** np.arange(-5, 0, 1)
    S = spdiags(e, 0, m, m) @ S
    S = S.astype(dtype)
    u, s, vh = svds(S, k, which='SM', solver=solver, maxiter=1000)
    c_svd = False
    _check_svds_n(S, k, u, s, vh, which='SM', check_svd=c_svd, atol=0.1)