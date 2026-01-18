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
@pytest.mark.parametrize('v0', ('hi', 1, np.ones(10, dtype=int)))
def test_svds_input_validation_v0_3(self, v0):
    A = np.ones((10, 10))
    message = '`v0` must be of floating or complex floating data type.'
    with pytest.raises(ValueError, match=message):
        svds(A, k=1, v0=v0, solver=self.solver)