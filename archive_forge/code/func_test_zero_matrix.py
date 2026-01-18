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
@pytest.mark.filterwarnings('ignore:k >= N - 1', reason='needed to demonstrate #16725')
@pytest.mark.parametrize('shape', ((3, 4), (4, 4), (4, 3), (4, 2)))
@pytest.mark.parametrize('dtype', (float, complex))
def test_zero_matrix(self, shape, dtype):
    k = 1
    n, m = shape
    A = np.zeros((n, m), dtype=dtype)
    if self.solver == 'arpack' and dtype is complex and (k == min(A.shape) - 1):
        pytest.skip('#16725')
    if self.solver == 'propack':
        pytest.skip('PROPACK failures unrelated to PR #16712')
    if self.solver == 'lobpcg':
        with pytest.warns(UserWarning, match='The problem size'):
            U, s, VH = svds(A, k, solver=self.solver)
    else:
        U, s, VH = svds(A, k, solver=self.solver)
    _check_svds(A, k, U, s, VH, check_usvh_A=True, check_svd=False)
    assert_array_equal(s, 0)