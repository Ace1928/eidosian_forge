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
@pytest.mark.parametrize('random_state', (0, 1, np.random.RandomState(0), np.random.default_rng(0)))
def test_svd_random_state_2(self, random_state):
    if self.solver == 'propack':
        if not has_propack:
            pytest.skip('PROPACK not available')
    n = 100
    k = 1
    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    random_state_2 = copy.deepcopy(random_state)
    res1a = svds(A, k, solver=self.solver, random_state=random_state)
    res2a = svds(A, k, solver=self.solver, random_state=random_state_2)
    for idx in range(3):
        assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
    _check_svds(A, k, *res1a)