import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
class TestBackward(BaseTest):
    """Test problem.backward() and problem.derivative()."""

    def setUp(self) -> None:
        try:
            import diffcp
            diffcp
        except ModuleNotFoundError:
            self.skipTest('diffcp not installed.')

    def test_scalar_quadratic(self) -> None:
        b = cp.Parameter()
        x = cp.Variable()
        quadratic = cp.square(x - 2 * b)
        problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])
        b.value = 3.0
        problem.solve(solver=cp.DIFFCP, requires_grad=True, eps=1e-10)
        self.assertAlmostEqual(x.value, 6.0)
        problem.backward()
        self.assertAlmostEqual(b.gradient, 2.0)
        x.gradient = 4.0
        problem.backward()
        self.assertAlmostEqual(b.gradient, 8.0)
        gradcheck(problem, atol=0.0001)
        perturbcheck(problem, atol=0.0001)
        problem.solve(solver=cp.DIFFCP, requires_grad=True, eps=1e-10)
        b.delta = 0.001
        problem.derivative()
        self.assertAlmostEqual(x.delta, 0.002)

    def test_l1_square(self) -> None:
        np.random.seed(0)
        n = 3
        x = cp.Variable(n)
        A = cp.Parameter((n, n))
        b = cp.Parameter(n, name='b')
        objective = cp.Minimize(cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective)
        self.assertTrue(problem.is_dpp())
        L = np.random.randn(n, n)
        A.value = L.T @ L + np.eye(n)
        b.value = np.random.randn(n)
        gradcheck(problem)
        perturbcheck(problem)

    def test_l1_rectangle(self) -> None:
        np.random.seed(0)
        m, n = (3, 2)
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m, name='b')
        objective = cp.Minimize(cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective)
        self.assertTrue(problem.is_dpp())
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        gradcheck(problem, atol=0.001)
        perturbcheck(problem, atol=0.001)

    def test_least_squares(self) -> None:
        np.random.seed(0)
        m, n = (20, 5)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(obj))
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        gradcheck(problem, solve_methods=[s.SCS])
        perturbcheck(problem, solve_methods=[s.SCS])

    def test_logistic_regression(self) -> None:
        np.random.seed(0)
        N, n = (5, 2)
        X_np = np.random.randn(N, n)
        a_true = np.random.randn(n, 1)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        y = np.round(sigmoid(X_np @ a_true + np.random.randn(N, 1) * 0.5))
        a = cp.Variable((n, 1))
        X = cp.Parameter((N, n))
        lam = cp.Parameter(nonneg=True)
        log_likelihood = cp.sum(cp.multiply(y, X @ a) - cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), X @ a]).T, axis=0, keepdims=True).T)
        problem = cp.Problem(cp.Minimize(-log_likelihood + lam * cp.sum_squares(a)))
        X.value = X_np
        lam.value = 1
        gradcheck(problem, solve_methods=[s.SCS], atol=0.1, eps=1e-08)
        perturbcheck(problem, solve_methods=[s.SCS], atol=0.0001)

    def test_entropy_maximization(self) -> None:
        np.random.seed(0)
        n, m, p = (5, 3, 2)
        tmp = np.random.rand(n)
        A_np = np.random.randn(m, n)
        b_np = A_np.dot(tmp)
        F_np = np.random.randn(p, n)
        g_np = F_np.dot(tmp) + np.random.rand(p)
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        F = cp.Parameter((p, n))
        g = cp.Parameter(p)
        obj = cp.Maximize(cp.sum(cp.entr(x)) - cp.sum_squares(x))
        constraints = [A @ x == b, F @ x <= g]
        problem = cp.Problem(obj, constraints)
        A.value = A_np
        b.value = b_np
        F.value = F_np
        g.value = g_np
        gradcheck(problem, solve_methods=[s.SCS], atol=0.01, eps=1e-08, max_iters=10000)
        perturbcheck(problem, solve_methods=[s.SCS], atol=0.0001)

    def test_lml(self) -> None:
        np.random.seed(0)
        k = 2
        x = cp.Parameter(4)
        y = cp.Variable(4)
        obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1.0 - y))
        cons = [cp.sum(y) == k]
        problem = cp.Problem(cp.Minimize(obj), cons)
        x.value = np.array([1.0, -1.0, -1.0, -1.0])
        gradcheck(problem, solve_methods=[s.SCS], atol=0.01)
        perturbcheck(problem, solve_methods=[s.SCS], atol=0.0001)

    def test_sdp(self) -> None:
        np.random.seed(0)
        n = 3
        p = 3
        C = cp.Parameter((n, n))
        As = [cp.Parameter((n, n)) for _ in range(p)]
        bs = [cp.Parameter((1, 1)) for _ in range(p)]
        C.value = np.random.randn(n, n)
        for A, b in zip(As, bs):
            A.value = np.random.randn(n, n)
            b.value = np.random.randn(1, 1)
        X = cp.Variable((n, n), PSD=True)
        constraints = [cp.trace(As[i] @ X) == bs[i] for i in range(p)]
        problem = cp.Problem(cp.Minimize(cp.trace(C @ X) + cp.sum_squares(X)), constraints)
        gradcheck(problem, solve_methods=[s.SCS], atol=0.001, eps=1e-10)
        perturbcheck(problem, solve_methods=[s.SCS])

    def test_forget_requires_grad(self) -> None:
        np.random.seed(0)
        m, n = (20, 5)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(obj))
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        problem.solve(cp.SCS)
        with self.assertRaisesRegex(ValueError, 'backward can only be called after calling solve with `requires_grad=True`'):
            problem.backward()
        with self.assertRaisesRegex(ValueError, 'derivative can only be called after calling solve with `requires_grad=True`'):
            problem.derivative()

    def test_infeasible(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
        param.value = 1
        problem.solve(solver=cp.DIFFCP, requires_grad=True)
        with self.assertRaisesRegex(cp.SolverError, 'Backpropagating through infeasible/unbounded.*'):
            problem.backward()
        with self.assertRaisesRegex(ValueError, 'Differentiating through infeasible/unbounded.*'):
            problem.derivative()

    def test_unbounded(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(x), [x <= param])
        param.value = 1
        problem.solve(solver=cp.DIFFCP, requires_grad=True)
        with self.assertRaisesRegex(cp.error.SolverError, 'Backpropagating through infeasible/unbounded.*'):
            problem.backward()
        with self.assertRaisesRegex(ValueError, 'Differentiating through infeasible/unbounded.*'):
            problem.derivative()

    def test_unsupported_solver(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(x), [x <= param])
        param.value = 1
        with self.assertRaisesRegex(ValueError, 'When requires_grad is True, the only supported solver is SCS.*'):
            problem.solve(cp.ECOS, requires_grad=True)

    def test_zero_in_problem_data(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        param.value = 0.0
        problem = cp.Problem(cp.Minimize(x), [param * x >= 0])
        data, _, _ = problem.get_problem_data(cp.DIFFCP)
        A = data[s.A]
        self.assertIn(0.0, A.data)