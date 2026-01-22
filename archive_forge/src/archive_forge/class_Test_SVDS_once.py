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
class Test_SVDS_once:

    @pytest.mark.parametrize('solver', ['ekki', object])
    def test_svds_input_validation_solver(self, solver):
        message = 'solver must be one of'
        with pytest.raises(ValueError, match=message):
            svds(np.ones((3, 4)), k=2, solver=solver)