import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
class TestSCS(BaseTest):
    """ Unit tests for SCS. """

    def setUp(self) -> None:
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(2, name='y')
        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def assertItemsAlmostEqual(self, a, b, places: int=2) -> None:
        super(TestSCS, self).assertItemsAlmostEqual(a, b, places=places)

    def assertAlmostEqual(self, a, b, places: int=2) -> None:
        super(TestSCS, self).assertAlmostEqual(a, b, places=places)

    def test_scs_retry(self) -> None:
        """Test that SCS retry doesn't trigger a crash.
        """
        n_sec = 20
        np.random.seed(315)
        mu = np.random.random(n_sec)
        random_mat = np.random.rand(n_sec, n_sec)
        C = np.dot(random_mat, random_mat.transpose())
        x = cp.Variable(n_sec)
        prob = cp.Problem(cp.Minimize(cp.QuadForm(x, C)), [cp.sum(x) == 1, 0 <= x, x <= 1, x @ mu >= np.max(mu) - 1e-06])
        prob.solve(cp.SCS)
        assert prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

    def test_scs_options(self) -> None:
        """Test that all the SCS solver options work.
        """
        EPS = 0.0001
        x = cp.Variable(2, name='x')
        prob = cp.Problem(cp.Minimize(cp.norm(x, 1) + 1.0), [x == 0])
        for i in range(2):
            prob.solve(solver=cp.SCS, max_iters=50, eps=EPS, alpha=1.2, verbose=True, normalize=True, use_indirect=False)
        self.assertAlmostEqual(prob.value, 1.0, places=2)
        self.assertItemsAlmostEqual(x.value, [0, 0], places=2)

    def test_log_problem(self) -> None:
        obj = cp.Maximize(cp.sum(cp.log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])
        obj = cp.Minimize(sum(self.x))
        constr = [cp.log(self.x) >= 0, self.x <= [1, 1]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])
        obj = cp.Maximize(cp.log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.SCS)

    def test_sigma_max(self) -> None:
        """Test sigma_max.
        """
        const = cp.Constant([[1, 2, 3], [4, 5, 6]])
        constr = [self.C == const]
        prob = cp.Problem(cp.Minimize(cp.norm(self.C, 2)), constr)
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, cp.norm(const, 2).value)
        self.assertItemsAlmostEqual(self.C.value, const.value)

    def test_sdp_var(self) -> None:
        """Test sdp var.
        """
        const = cp.Constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X = cp.Variable((3, 3), PSD=True)
        prob = cp.Problem(cp.Minimize(0), [X == const])
        prob.solve(solver=cp.SCS)
        self.assertEqual(prob.status, cp.INFEASIBLE)

    def test_complex_matrices(self) -> None:
        """Test complex matrices.
        """
        np.random.seed(0)
        K = np.array(np.random.rand(2, 2) + 1j * np.random.rand(2, 2))
        n1 = la.svdvals(K).sum()
        X = cp.Variable((2, 2), complex=True)
        Y = cp.Variable((2, 2), complex=True)
        objective = cp.Minimize(cp.real(0.5 * cp.trace(X) + 0.5 * cp.trace(Y)))
        constraints = [cp.bmat([[X, -K.conj().T], [-K, Y]]) >> 0, X >> 0, Y >> 0]
        problem = cp.Problem(objective, constraints)
        sol_scs = problem.solve(solver='SCS')
        self.assertEqual(constraints[0].dual_value.shape, (4, 4))
        self.assertEqual(constraints[1].dual_value.shape, (2, 2))
        self.assertEqual(constraints[2].dual_value.shape, (2, 2))
        self.assertAlmostEqual(sol_scs, n1)

    def test_entr(self) -> None:
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.entr(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n])

    def test_exp(self) -> None:
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Minimize(cp.sum(cp.exp(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n])

    def test_log(self) -> None:
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.log(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n * [1.0 / n])

    def test_solve_problem_twice(self) -> None:
        """Test a problem with log.
        """
        n = 5
        x = cp.Variable(n)
        obj = cp.Maximize(cp.sum(cp.log(x)))
        p = cp.Problem(obj, [cp.sum(x) == 1])
        p.solve(solver=cp.SCS)
        first_value = x.value
        self.assertItemsAlmostEqual(first_value, n * [1.0 / n])
        p.solve(solver=cp.SCS)
        second_value = x.value
        self.assertItemsAlmostEqual(first_value, second_value)

    def test_warm_start(self) -> None:
        """Test warm starting.
        """
        x = cp.Variable(10)
        obj = cp.Minimize(cp.sum(cp.exp(x)))
        prob = cp.Problem(obj, [cp.sum(x) == 1])
        result = prob.solve(solver=cp.SCS)
        time = prob.solver_stats.solve_time
        result2 = prob.solve(solver=cp.SCS, warm_start=True)
        time2 = prob.solver_stats.solve_time
        self.assertAlmostEqual(result2, result, places=2)
        print(time > time2)

    def test_warm_start_diffcp(self) -> None:
        """Test warm starting in diffcvx.
        """
        try:
            import diffcp
            diffcp
        except ModuleNotFoundError:
            self.skipTest('diffcp not installed.')
        x = cp.Variable(10)
        obj = cp.Minimize(cp.sum(cp.exp(x)))
        prob = cp.Problem(obj, [cp.sum(x) == 1])
        result = prob.solve(solver=cp.DIFFCP)
        result2 = prob.solve(solver=cp.DIFFCP, warm_start=True)
        self.assertAlmostEqual(result2, result, places=2)

    def test_psd_constraint(self) -> None:
        """Test PSD constraint.
        """
        s = cp.Variable((2, 2))
        obj = cp.Maximize(cp.minimum(s[0, 1], 10))
        const = [s >> 0, cp.diag(s) == np.ones(2)]
        prob = cp.Problem(obj, const)
        r = prob.solve(solver=cp.SCS)
        s = s.value
        print(const[0].residual)
        print('value', r)
        print('s', s)
        print('eigs', np.linalg.eig(s + s.T)[0])
        eigs = np.linalg.eig(s + s.T)[0]
        self.assertEqual(np.all(eigs >= 0), True)

    def test_quad_obj(self) -> None:
        """Test SCS canonicalization with a quadratic objective.
        """
        import scs
        if Version(scs.__version__) >= Version('3.0.0'):
            x = cp.Variable(2)
            expr = cp.sum_squares(x)
            constr = [x >= 1]
            prob = cp.Problem(cp.Minimize(expr), constr)
            data = prob.get_problem_data(solver=cp.SCS)
            self.assertItemsAlmostEqual(data[0]['P'].A, 2 * np.eye(2))
            solution1 = prob.solve(solver=cp.SCS)
            prob = cp.Problem(cp.Minimize(expr), constr)
            solver_opts = {'use_quad_obj': False}
            data = prob.get_problem_data(solver=cp.SCS, solver_opts=solver_opts)
            assert 'P' not in data[0]
            solution2 = prob.solve(solver=cp.SCS, **solver_opts)
            assert np.isclose(solution1, solution2)
            expr = cp.norm(x, 1)
            prob = cp.Problem(cp.Minimize(expr), constr)
            data = prob.get_problem_data(solver=cp.SCS)
            assert 'P' not in data[0]

    def test_quad_obj_with_power(self) -> None:
        """Test a mixed quadratic/power objective.
        """
        import scs
        if Version(scs.__version__) >= Version('3.0.0'):
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(x ** 1.6 + x ** 2), [x >= 1])
            prob.solve(solver=cp.SCS, use_quad_obj=True)
            self.assertAlmostEqual(prob.value, 2)
            self.assertAlmostEqual(x.value, 1)
            data = prob.get_problem_data(solver=cp.SCS, solver_opts={'use_quad_obj': True})
            assert 'P' in data[0]
            assert data[0]['dims'].soc

    def test_scs_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='SCS')

    def test_scs_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='SCS')

    def test_scs_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='SCS', eps=1e-06)

    def test_scs_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='SCS', eps=1e-06)

    def test_scs_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='SCS')
        StandardTestSOCPs.test_socp_3ax1(solver='SCS')

    def test_scs_sdp_1min(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='SCS')

    def test_scs_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='SCS', eps=1e-05)

    def test_scs_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='SCS', eps=1e-05)

    def test_scs_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='SCS', eps=1e-05)

    def test_scs_sdp_pcp_1(self):
        StandardTestMixedCPs.test_sdp_pcp_1(solver='SCS')

    def test_scs_pcp_1(self) -> None:
        StandardTestPCPs.test_pcp_1(solver='SCS')

    def test_scs_pcp_2(self) -> None:
        StandardTestPCPs.test_pcp_2(solver='SCS')

    def test_scs_pcp_3(self) -> None:
        StandardTestPCPs.test_pcp_3(solver='SCS', eps=1e-12)