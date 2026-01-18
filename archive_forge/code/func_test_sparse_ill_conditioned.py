import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_sparse_ill_conditioned(self):
    data = np.array([1.0, 1.0, 1.0, 1.0 + 1e-06, 1.0])
    row = np.array([0, 0, 1, 2, 2])
    col = np.array([0, 2, 1, 0, 2])
    A = coo_matrix((data, (row, col)), shape=(3, 3))
    exact_sol = lsq_linear(A.toarray(), b, lsq_solver='exact')
    default_lsmr_sol = lsq_linear(A, b, lsq_solver='lsmr')
    with pytest.raises(AssertionError, match=''):
        assert_allclose(exact_sol.x, default_lsmr_sol.x)
    conv_lsmr = lsq_linear(A, b, lsq_solver='lsmr', lsmr_maxiter=10)
    assert_allclose(exact_sol.x, conv_lsmr.x)