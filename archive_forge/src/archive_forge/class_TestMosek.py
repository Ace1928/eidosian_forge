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
@unittest.skipUnless('MOSEK' in INSTALLED_SOLVERS, 'MOSEK is not installed.')
class TestMosek(unittest.TestCase):

    def test_mosek_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='MOSEK')

    def test_mosek_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='MOSEK')
        StandardTestLPs.test_lp_1(solver='MOSEK', places=6, bfs=True)

    def test_mosek_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='MOSEK')

    def test_mosek_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='MOSEK')

    def test_mosek_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='MOSEK')

    def test_mosek_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='MOSEK')

    def test_mosek_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='MOSEK')

    def test_mosek_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='MOSEK')

    def test_mosek_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='MOSEK')

    def test_mosek_socp_3(self) -> None:
        StandardTestSOCPs.test_socp_3ax0(solver='MOSEK')
        StandardTestSOCPs.test_socp_3ax1(solver='MOSEK')

    def test_mosek_sdp_1(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='MOSEK')
        StandardTestSDPs.test_sdp_1max(solver='MOSEK')

    def test_mosek_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='MOSEK')

    def test_mosek_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='MOSEK')

    def test_mosek_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='MOSEK')

    def test_mosek_pcp_1(self) -> None:
        StandardTestPCPs.test_pcp_1(solver='MOSEK', places=2)

    def test_mosek_pcp_2(self) -> None:
        StandardTestPCPs.test_pcp_2(solver='MOSEK')

    def test_mosek_pcp_3(self) -> None:
        StandardTestPCPs.test_pcp_3(solver='MOSEK')

    def test_mosek_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='MOSEK')

    def test_mosek_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='MOSEK')

    def test_mosek_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='MOSEK')

    def test_mosek_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='MOSEK')

    def test_mosek_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='MOSEK')

    def test_mosek_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='MOSEK', places=3)

    def test_mosek_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='MOSEK')

    def test_mosek_mi_pcp_0(self) -> None:
        StandardTestPCPs.test_mi_pcp_0(solver='MOSEK')

    def test_mosek_params(self) -> None:
        import mosek
        n = 10
        m = 4
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)
        invalid_mosek_params = {'MSK_IPAR_NUM_THREADS': '11.3'}
        with self.assertRaises(mosek.Error):
            problem.solve(solver=cp.MOSEK, mosek_params=invalid_mosek_params)
        with self.assertRaises(ValueError):
            problem.solve(solver=cp.MOSEK, invalid_kwarg=None)
        mosek_params = {mosek.dparam.basis_tol_x: 1e-08, 'MSK_IPAR_INTPNT_MAX_ITERATIONS': 20, 'MSK_IPAR_NUM_THREADS': '17', 'MSK_IPAR_PRESOLVE_USE': 'MSK_PRESOLVE_MODE_OFF', 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-09, 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': '1e-9'}
        with pytest.warns():
            problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)

    def test_mosek_simplex(self) -> None:
        n = 10
        m = 4
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, mosek_params={'MSK_IPAR_OPTIMIZER': 'MSK_OPTIMIZER_DUAL_SIMPLEX'})

    def test_mosek_iis(self) -> None:
        """Test IIS feature in Mosek."""
        n = 2
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum(x))
        constraints = [x[0] >= 1, x[0] <= -1, x[1] >= 3]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        iis = problem.solver_stats.extra_stats['IIS']
        assert iis[constraints[0].id] > 0
        assert iis[constraints[1].id] > 0
        assert iis[constraints[2].id] == 0
        n = 3
        m = 2
        X = cp.Variable((m, n))
        y = cp.Variable()
        objective = cp.Minimize(cp.sum(X))
        constraints = [y == 2, X >= 3, X[0, 0] + y <= -5]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        iis = problem.solver_stats.extra_stats['IIS']
        assert abs(iis[constraints[0].id]) > 0
        dual1 = np.reshape(iis[constraints[1].id], X.shape, order='C')
        assert dual1[0, 0] > 0
        assert dual1[0, 1] == 0
        assert np.all(dual1[1, :] == 0)
        assert iis[constraints[2].id] > 0

    def test_mosek_sdp_power(self) -> None:
        """Test the problem in issue #2128"""
        StandardTestMixedCPs.test_sdp_pcp_1(solver='MOSEK')

    def test_power_portfolio(self) -> None:
        """Test the portfolio problem in issue #2042"""
        T, N = (200, 10)
        rs = np.random.RandomState(123)
        mean = np.zeros(N) + 1 / 1000
        cov = rs.rand(N, N) * 1.5 - 0.5
        cov = cov @ cov.T / 1000 + np.diag(rs.rand(N) * 0.7 + 0.3) / 1000
        Y = st.multivariate_normal.rvs(mean=mean, cov=cov, size=T, random_state=rs)
        w = cp.Variable((N, 1))
        t = cp.Variable((1, 1))
        z = cp.Variable((1, 1))
        omega = cp.Variable((T, 1))
        psi = cp.Variable((T, 1))
        nu = cp.Variable((T, 1))
        epsilon = cp.Variable((T, 1))
        k = cp.Variable((1, 1))
        b = np.ones((1, N)) / N
        X = Y @ w
        h = 0.2
        ones = np.ones((T, 1))
        constraints = [cp.constraints.power.PowCone3D(z * (1 + h) / (2 * h) * ones, psi * (1 + h) / h, epsilon, 1 / (1 + h)), cp.constraints.power.PowCone3D(omega / (1 - h), nu / h, -z / (2 * h) * ones, 1 - h), -X - t + epsilon + omega <= 0, w >= 0, z >= 0]
        obj = t + z + cp.sum(psi + nu)
        constraints += [cp.sum(w) == k, k >= 0, b @ cp.log(w) >= 1]
        objective = cp.Minimize(obj)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK)
        assert prob.status is cp.OPTIMAL

    def test_mosek_accept_unknown(self) -> None:
        mosek_param = {'MSK_IPAR_INTPNT_MAX_ITERATIONS': 0}
        sth = sths.lp_5()
        sth.solve(solver=cp.MOSEK, accept_unknown=True, mosek_params=mosek_param)
        assert sth.prob.status in {cp.OPTIMAL_INACCURATE, cp.OPTIMAL}
        with pytest.raises(cp.error.SolverError, match="Solver 'MOSEK' failed"):
            sth.solve(solver=cp.MOSEK, mosek_params=mosek_param)

    def test_mosek_number_iters(self) -> None:
        sth = sths.lp_5()
        sth.solve(solver=cp.MOSEK)
        assert sth.prob.solver_stats.num_iters >= 0
        assert sth.prob.solver_stats.extra_stats['mio_intpnt_iter'] == 0
        assert sth.prob.solver_stats.extra_stats['mio_simplex_iter'] == 0

    def test_eps_keyword(self) -> None:
        """Test that the eps keyword is accepted"""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve(solver=cp.MOSEK, eps=1e-08, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-06})
        assert prob.status is cp.OPTIMAL
        import mosek
        with pytest.raises(mosek.Error, match='The parameter value 0.1 is too large'):
            prob.solve(solver=cp.MOSEK, eps=0.1, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-06})
        from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK
        all_params = MOSEK.tolerance_params()
        prob.solve(solver=cp.MOSEK, eps=0.1, mosek_params={p: 1e-06 for p in all_params})
        assert prob.status is cp.OPTIMAL
        with pytest.raises(AssertionError, match='not compatible'):
            prob.solve(solver=cp.MOSEK, eps=0.1, mosek_params={mosek.dparam.intpnt_co_tol_dfeas: 1e-06})