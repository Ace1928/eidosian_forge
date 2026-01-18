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
@pytest.mark.parametrize('transpose', (True, False))
@pytest.mark.parametrize('n', range(4, 9))
def test_svds_input_validation_v0_1(self, transpose, n):
    rng = np.random.default_rng(0)
    A = rng.random((5, 7))
    v0 = rng.random(n)
    if transpose:
        A = A.T
    k = 2
    message = '`v0` must have shape'
    required_length = A.shape[0] if self.solver == 'propack' else min(A.shape)
    if n != required_length:
        with pytest.raises(ValueError, match=message):
            svds(A, k=k, v0=v0, solver=self.solver)