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
@pytest.mark.parametrize('tol', ([], 'hi'))
def test_svds_input_validation_tol_2(self, tol):
    message = "'<' not supported between instances"
    with pytest.raises(TypeError, match=message):
        svds(np.eye(10), tol=tol, solver=self.solver)