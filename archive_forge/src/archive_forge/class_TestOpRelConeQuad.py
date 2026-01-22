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
@unittest.skipUnless(sdp_ipm_installed(), 'First-order solvers are too slow for the accuracy we need.')
class TestOpRelConeQuad(BaseTest):

    def setUp(self, n=3) -> None:
        self.n = n
        self.a = cp.Variable(shape=(n,), pos=True)
        self.b = cp.Variable(shape=(n,), pos=True)
        if hasattr(np.random, 'default_rng'):
            self.rng = np.random.default_rng(0)
        else:
            self.rng = np.random.RandomState(0)
        if hasattr(self.rng, 'random'):
            rand_gen_func = self.rng.random
        else:
            rand_gen_func = self.rng.random_sample
        self.a_lower = np.cumsum(rand_gen_func(n))
        self.a_upper = self.a_lower + 0.05 * rand_gen_func(n)
        self.b_lower = np.cumsum(rand_gen_func(n))
        self.b_upper = self.b_lower + 0.05 * rand_gen_func(n)
        self.base_cons = [self.a_lower <= self.a, self.a <= self.a_upper, self.b_lower <= self.b, self.b <= self.b_upper]
        installed_solvers = cp.installed_solvers()
        if cp.MOSEK in installed_solvers:
            self.solver = cp.MOSEK
        elif cp.CVXOPT in installed_solvers:
            self.solver = cp.CVXOPT
        elif cp.COPT in installed_solvers:
            self.solver = cp.COPT
        else:
            raise RuntimeError('No viable solver installed.')
        pass

    @staticmethod
    def Dop_commute(a: np.ndarray, b: np.ndarray, U: np.ndarray):
        D = np.diag(a * np.log(a / b))
        if np.iscomplexobj(U):
            out = U @ D @ U.conj().T
        else:
            out = U @ D @ U.T
        return out

    @staticmethod
    def sum_rel_entr_approx(a: cp.Expression, b: cp.Expression, apx_m: int, apx_k: int):
        n = a.size
        assert n == b.size
        epi_vec = cp.Variable(shape=n)
        con = cp.constraints.RelEntrConeQuad(a, b, epi_vec, apx_m, apx_k)
        objective = cp.Minimize(cp.sum(epi_vec))
        return (objective, con)

    def oprelcone_1(self, apx_m, apx_k, real) -> STH.SolverTestHelper:
        """
        These tests construct two matrices that commute (imposing all eigenvectors equal)
        and then use the fact that: T=Dop(A, B) for (A, B, T) in OpRelEntrConeQuad
        i.e. T >> Dop(A, B) for an objective that is an increasing function of the
        eigenvalues (which we here take to be the trace), we compute the reference
        objective value as tr(Dop) whose correctness can be seen by writing out
        tr(T)=tr(T-Dop)+tr(Dop), where tr(T-Dop)>=0 because of PSD-ness of (T-Dop),
        and at optimality we have (T-Dop)=0 (the zero matrix of corresponding size)
        For the case that the input matrices commute, Dop takes on a particularly
        simplified form, i.e.: U @ diag(a * log(a/b)) @ U^{-1} (which is implemented
        in the Dop_commute method above)
        """
        temp_obj, temp_con = TestOpRelConeQuad.sum_rel_entr_approx(self.a, self.b, apx_m, apx_k)
        temp_constraints = [con for con in self.base_cons]
        temp_constraints.append(temp_con)
        temp_prob = cp.Problem(temp_obj, temp_constraints)
        temp_prob.solve()
        expect_a = self.a.value
        expect_b = self.b.value
        expect_objective = temp_obj.value
        n = self.n
        if real:
            randmat = self.rng.normal(size=(n, n))
            U = sp.linalg.qr(randmat)[0]
            A = cp.symmetric_wrap(U @ cp.diag(self.a) @ U.T)
            B = cp.symmetric_wrap(U @ cp.diag(self.b) @ U.T)
            T = cp.Variable(shape=(n, n), symmetric=True)
        else:
            randmat = 1j * self.rng.normal(size=(n, n))
            randmat += self.rng.normal(size=(n, n))
            U = sp.linalg.qr(randmat)[0]
            A = cp.hermitian_wrap(U @ cp.diag(self.a) @ U.conj().T)
            B = cp.hermitian_wrap(U @ cp.diag(self.b) @ U.conj().T)
            T = cp.Variable(shape=(n, n), hermitian=True)
        main_con = cp.constraints.OpRelEntrConeQuad(A, B, T, apx_m, apx_k)
        obj = cp.Minimize(trace(T))
        expect_T = TestOpRelConeQuad.Dop_commute(expect_a, expect_b, U)
        con_pairs = [(con, None) for con in self.base_cons]
        con_pairs.append((main_con, None))
        obj_pair = (obj, expect_objective)
        var_pairs = [(T, expect_T)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_oprelcone_1_m1_k3_real(self):
        sth = self.oprelcone_1(1, 3, True)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m3_k1_real(self):
        sth = self.oprelcone_1(3, 1, True)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m4_k4_real(self):
        sth = self.oprelcone_1(4, 4, True)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m1_k3_complex(self):
        sth = self.oprelcone_1(1, 3, False)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m3_k1_complex(self):
        sth = self.oprelcone_1(3, 1, False)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def oprelcone_2(self) -> STH.SolverTestHelper:
        """
        This test uses the same idea from the tests with commutative matrices,
        instead, here, we make the input matrices to Dop, non-commutative,
        the same condition as before i.e. T=Dop(A, B) for (A, B, T) in OpRelEntrConeQuad
        (for an objective that is an increasing function of the eigenvalues) holds,
        the difference here then, is in how we compute the reference values, which
        has been done by assuming correctness of the original CVXQUAD matlab implementation
        """
        n, m, k = (4, 3, 3)
        U1 = np.array([[-0.05878522, -0.78378355, -0.49418311, -0.37149791], [0.67696027, -0.25733435, 0.59263364, -0.35254672], [0.43478177, 0.53648704, -0.54593428, -0.47444939], [0.59096015, -0.17788771, -0.32638042, 0.71595942]])
        U2 = np.array([[-0.42499169, 0.6887562, 0.55846178, 0.18198188], [-0.55478633, -0.7091174, 0.3884544, 0.19613213], [-0.55591804, 0.14358541, -0.72444644, 0.38146522], [0.4500548, -0.04637494, 0.11135968, 0.88481584]])
        a_diag = cp.Variable(shape=(n,), pos=True)
        b_diag = cp.Variable(shape=(n,), pos=True)
        A = U1 @ cp.diag(a_diag) @ U1.T
        B = U2 @ cp.diag(b_diag) @ U2.T
        T = cp.Variable(shape=(n, n), symmetric=True)
        a_lower = np.array([0.40683013, 1.34514597, 1.60057343, 2.13373667])
        a_upper = np.array([1.36158501, 1.61289351, 1.85065805, 3.06140939])
        b_lower = np.array([0.06858235, 0.36798274, 0.95956627, 1.16286541])
        b_upper = np.array([0.70446555, 1.16635299, 1.46126732, 1.81367755])
        con1 = cp.constraints.OpRelEntrConeQuad(A, B, T, m, k)
        con2 = a_lower <= a_diag
        con3 = a_diag <= a_upper
        con4 = b_lower <= b_diag
        con5 = b_diag <= b_upper
        con_pairs = [(con1, None), (con2, None), (con3, None), (con4, None), (con5, None)]
        obj = cp.Minimize(trace(T))
        expect_obj = 1.85476
        expect_T = np.array([[0.49316819, 0.20845265, 0.60474713, -0.5820242], [0.20845265, 0.31084053, 0.2264112, -0.8442255], [0.60474713, 0.2264112, 0.4687153, -0.85667283], [-0.5820242, -0.8442255, -0.85667283, 0.58206723]])
        obj_pair = (obj, expect_obj)
        var_pairs = [(T, expect_T)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_oprelcone_2(self):
        sth = self.oprelcone_2()
        sth.solve(self.solver)
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)