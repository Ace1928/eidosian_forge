import numpy as np
import cvxpy as cp
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_cone2cone import TestPowND
class TestKKT_PCPs(BaseTest):

    @staticmethod
    def non_vec_pow_nd() -> STH.SolverTestHelper:
        n_buyer = 4
        n_items = 6
        z = cp.Variable(shape=(2,))
        np.random.seed(0)
        V = np.random.rand(n_buyer, n_items)
        X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
        u = cp.sum(cp.multiply(V, X), axis=1)
        alpha1 = np.array([0.4069713, 0.10067042, 0.30507361, 0.18728467])
        alpha2 = np.array([0.13209105, 0.18918836, 0.36087677, 0.31784382])
        cons = [cp.PowConeND(u, z[0], alpha1), cp.PowConeND(u, z[1], alpha2), X >= 0, cp.sum(X, axis=0) <= 1]
        obj = cp.Maximize(z[0] + z[1])
        obj_pair = (obj, 2.415600275720486)
        var_pairs = [(X, None), (u, None), (z, None)]
        cons_pairs = [(con, None) for con in cons]
        return STH.SolverTestHelper(obj_pair, var_pairs, cons_pairs)

    @staticmethod
    def vec_pow_nd() -> STH.SolverTestHelper:
        n_buyer = 4
        n_items = 6
        z = cp.Variable(shape=(2,))
        np.random.seed(1)
        V = np.random.rand(n_buyer, n_items)
        X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
        u = cp.sum(cp.multiply(V, X), axis=1)
        alpha1 = np.array([0.02999541, 0.24340343, 0.03687151, 0.68972966])
        alpha2 = np.array([0.24041855, 0.1745123, 0.10012628, 0.48494287])
        cons = [cp.PowConeND(cp.vstack([u, u]), z, np.vstack([alpha1, alpha2]), axis=1), X >= 0, cp.sum(X, axis=0) <= 1]
        obj = cp.Maximize(z[0] + z[1])
        prob = cp.Problem(obj, cons)
        prob.solve(solver='SCS')
        obj_pair = (obj, 2.7003780870341516)
        cons_pairs = [(con, None) for con in cons]
        var_pairs = [(z, None), (X, None), (u, None)]
        return STH.SolverTestHelper(obj_pair, var_pairs, cons_pairs)

    def test_pcp_1(self, places: int=4):
        sth = STH.pcp_1()
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_2(self, places: int=4):
        sth = STH.pcp_2()
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_3(self, places: int=4):
        sth = STH.pcp_3()
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_4(self, places: int=3):
        sth = self.non_vec_pow_nd()
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_5(self, places: int=3):
        sth = self.vec_pow_nd()
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_primal_feasibility(places)
        sth.check_complementarity(places)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth

    def test_pcp_6(self, places: int=3):
        sth = TestPowND.pcp_4()
        sth.solve(solver='SCS', eps=1e-06)
        sth.check_dual_domains(places)
        sth.check_stationary_lagrangian(places)
        return sth