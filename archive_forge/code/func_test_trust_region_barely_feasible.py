import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
def test_trust_region_barely_feasible(self):
    H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
    A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
    c = np.array([-2, -3, -3, 1])
    b = -np.array([3, 0])
    trust_radius = 2.32379000772445
    Z, _, Y = projections(A)
    x, info = projected_cg(H, c, Z, Y, b, tol=0, trust_radius=trust_radius)
    assert_equal(info['stop_cond'], 2)
    assert_equal(info['hits_boundary'], True)
    assert_array_almost_equal(np.linalg.norm(x), trust_radius)
    assert_array_almost_equal(x, -Y.dot(b))