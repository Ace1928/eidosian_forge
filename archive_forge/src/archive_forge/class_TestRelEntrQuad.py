import unittest
import numpy as np
import pytest
import scipy as sp
import cvxpy as cp
from cvxpy import settings as s
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as INSTALLED_MI
from cvxpy.reductions.solvers.defines import MI_SOCP_SOLVERS as MI_SOCP
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest
class TestRelEntrQuad(BaseTest):

    def expcone_1(self) -> STH.SolverTestHelper:
        """
        min   3 * x[0] + 2 * x[1] + x[2]
        s.t.  0.1 <= x[0] + x[1] + x[2] <= 1
              x >= 0
              and ...
                x[0] >= x[1] * exp(x[2] / x[1])
              equivalently ...
                x[0] / x[1] >= exp(x[2] / x[1])
                log(x[0] / x[1]) >= x[2] / x[1]
                x[1] log(x[1] / x[0]) <= -x[2]
        """
        x = cp.Variable(shape=(3, 1))
        cone_con = ExpCone(x[2], x[1], x[0]).as_quad_approx(5, 5)
        constraints = [cp.sum(x) <= 1.0, cp.sum(x) >= 0.1, x >= 0, cone_con]
        obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
        obj_pair = (obj, 0.23534820622420757)
        expect_exp = [np.array([-1.35348213]), np.array([-0.35348211]), np.array([0.64651792])]
        con_pairs = [(constraints[0], 0), (constraints[1], 2.3534821130067614), (constraints[2], np.zeros(shape=(3, 1))), (constraints[3], expect_exp)]
        expect_x = np.array([[0.05462721], [0.02609378], [0.01927901]])
        var_pairs = [(x, expect_x)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_expcone_1(self):
        sth = self.expcone_1()
        sth.solve(solver='ECOS')
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)

    def expcone_socp_1(self) -> STH.SolverTestHelper:
        """
        A random risk-parity portfolio optimization problem.
        """
        sigma = np.array([[1.83, 1.79, 3.22], [1.79, 2.18, 3.18], [3.22, 3.18, 8.69]])
        L = np.linalg.cholesky(sigma)
        c = 0.75
        t = cp.Variable(name='t')
        x = cp.Variable(shape=(3,), name='x')
        s = cp.Variable(shape=(3,), name='s')
        e = cp.Constant(np.ones(3))
        objective = cp.Minimize(t - c * e @ s)
        con1 = cp.norm(L.T @ x, p=2) <= t
        con2 = ExpCone(s, e, x).as_quad_approx(5, 5)
        obj_pair = (objective, 4.0751197)
        var_pairs = [(x, np.array([0.57608346, 0.54315695, 0.28037716])), (s, np.array([-0.5515, -0.61036, -1.27161]))]
        con_pairs = [(con1, 1.0), (con2, [None, None, None])]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_expcone_socp_1(self):
        sth = self.expcone_socp_1()
        sth.solve(solver='ECOS')
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)