import numpy as np
from scipy.sparse import csc_matrix
from scipy.optimize._trustregion_constr.qp_subproblem \
from scipy.optimize._trustregion_constr.projections \
from numpy.testing import TestCase, assert_array_almost_equal, assert_equal
import pytest
class TestProjectCG(TestCase):

    def test_nocedal_example(self):
        H = csc_matrix([[6, 2, 1], [2, 5, 2], [1, 2, 4]])
        A = csc_matrix([[1, 0, 1], [0, 1, 1]])
        c = np.array([-8, -3, -3])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b)
        assert_equal(info['stop_cond'], 4)
        assert_equal(info['hits_boundary'], False)
        assert_array_almost_equal(x, [2, -1, 1])

    def test_compare_with_direct_fact(self):
        H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b, tol=0)
        x_kkt, _ = eqp_kktfact(H, c, A, b)
        assert_equal(info['stop_cond'], 1)
        assert_equal(info['hits_boundary'], False)
        assert_array_almost_equal(x, x_kkt)

    def test_trust_region_infeasible(self):
        H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 1
        Z, _, Y = projections(A)
        with pytest.raises(ValueError):
            projected_cg(H, c, Z, Y, b, trust_radius=trust_radius)

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

    def test_hits_boundary(self):
        H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 3
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b, tol=0, trust_radius=trust_radius)
        assert_equal(info['stop_cond'], 2)
        assert_equal(info['hits_boundary'], True)
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)

    def test_negative_curvature_unconstrained(self):
        H = csc_matrix([[1, 2, 1, 3], [2, 0, 2, 4], [1, 2, 0, 2], [3, 4, 2, 0]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 0, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        with pytest.raises(ValueError):
            projected_cg(H, c, Z, Y, b, tol=0)

    def test_negative_curvature(self):
        H = csc_matrix([[1, 2, 1, 3], [2, 0, 2, 4], [1, 2, 0, 2], [3, 4, 2, 0]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 0, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        trust_radius = 1000
        x, info = projected_cg(H, c, Z, Y, b, tol=0, trust_radius=trust_radius)
        assert_equal(info['stop_cond'], 3)
        assert_equal(info['hits_boundary'], True)
        assert_array_almost_equal(np.linalg.norm(x), trust_radius)

    def test_inactive_box_constraints(self):
        H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b, tol=0, lb=[0.5, -np.inf, -np.inf, -np.inf], return_all=True)
        x_kkt, _ = eqp_kktfact(H, c, A, b)
        assert_equal(info['stop_cond'], 1)
        assert_equal(info['hits_boundary'], False)
        assert_array_almost_equal(x, x_kkt)

    def test_active_box_constraints_maximum_iterations_reached(self):
        H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b, tol=0, lb=[0.8, -np.inf, -np.inf, -np.inf], return_all=True)
        assert_equal(info['stop_cond'], 1)
        assert_equal(info['hits_boundary'], True)
        assert_array_almost_equal(A.dot(x), -b)
        assert_array_almost_equal(x[0], 0.8)

    def test_active_box_constraints_hits_boundaries(self):
        H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 3
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b, tol=0, ub=[np.inf, np.inf, 1.6, np.inf], trust_radius=trust_radius, return_all=True)
        assert_equal(info['stop_cond'], 2)
        assert_equal(info['hits_boundary'], True)
        assert_array_almost_equal(x[2], 1.6)

    def test_active_box_constraints_hits_boundaries_infeasible_iter(self):
        H = csc_matrix([[6, 2, 1, 3], [2, 5, 2, 4], [1, 2, 4, 5], [3, 4, 5, 7]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 1, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        trust_radius = 4
        Z, _, Y = projections(A)
        x, info = projected_cg(H, c, Z, Y, b, tol=0, ub=[np.inf, 0.1, np.inf, np.inf], trust_radius=trust_radius, return_all=True)
        assert_equal(info['stop_cond'], 2)
        assert_equal(info['hits_boundary'], True)
        assert_array_almost_equal(x[1], 0.1)

    def test_active_box_constraints_negative_curvature(self):
        H = csc_matrix([[1, 2, 1, 3], [2, 0, 2, 4], [1, 2, 0, 2], [3, 4, 2, 0]])
        A = csc_matrix([[1, 0, 1, 0], [0, 1, 0, 1]])
        c = np.array([-2, -3, -3, 1])
        b = -np.array([3, 0])
        Z, _, Y = projections(A)
        trust_radius = 1000
        x, info = projected_cg(H, c, Z, Y, b, tol=0, ub=[np.inf, np.inf, 100, np.inf], trust_radius=trust_radius)
        assert_equal(info['stop_cond'], 3)
        assert_equal(info['hits_boundary'], True)
        assert_array_almost_equal(x[2], 100)