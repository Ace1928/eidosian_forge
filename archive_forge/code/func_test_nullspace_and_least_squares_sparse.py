import numpy as np
import scipy.linalg
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import (TestCase, assert_array_almost_equal,
def test_nullspace_and_least_squares_sparse(self):
    A_dense = np.array([[1, 2, 3, 4, 0, 5, 0, 7], [0, 8, 7, 0, 1, 5, 9, 0], [1, 0, 0, 0, 0, 1, 2, 3]])
    At_dense = A_dense.T
    A = csc_matrix(A_dense)
    test_points = ([1, 2, 3, 4, 5, 6, 7, 8], [1, 10, 3, 0, 1, 6, 7, 8], [1.12, 10, 0, 0, 100000, 6, 0.7, 8])
    for method in available_sparse_methods:
        Z, LS, _ = projections(A, method)
        for z in test_points:
            x = Z.matvec(z)
            assert_array_almost_equal(A.dot(x), 0)
            assert_array_almost_equal(orthogonality(A, x), 0)
            x = LS.matvec(z)
            x2 = scipy.linalg.lstsq(At_dense, z)[0]
            assert_array_almost_equal(x, x2)