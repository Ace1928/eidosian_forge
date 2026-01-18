import numpy as np
import scipy.sparse as sp
import cvxpy
from cvxpy.tests.base_test import BaseTest
def test_sparse_constant_not_allowed(self) -> None:
    sparse_matrix = cvxpy.Constant(sp.csc_matrix(np.array([1.0, 2.0])))
    self.assertFalse(sparse_matrix.is_log_log_constant())