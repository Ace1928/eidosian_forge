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
@pytest.mark.parametrize('ncv', list(range(-1, 8)) + [4.5, '5'])
def test_svds_input_validation_ncv_1(self, ncv):
    rng = np.random.default_rng(0)
    A = rng.random((6, 7))
    k = 3
    if ncv in {4, 5}:
        u, s, vh = svds(A, k=k, ncv=ncv, solver=self.solver)
        _check_svds(A, k, u, s, vh)
    else:
        message = '`ncv` must be an integer satisfying'
        with pytest.raises(ValueError, match=message):
            svds(A, k=k, ncv=ncv, solver=self.solver)