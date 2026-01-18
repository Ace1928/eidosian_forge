import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_param_used_twice(self) -> None:
    alpha = cp.Parameter(pos=True)
    x = cp.Variable(pos=True)
    objective = cp.Maximize(x)
    constr = [cp.one_minus_pos(x) >= alpha ** 2 + alpha ** 3]
    problem = cp.Problem(objective, constr)
    alpha.value = 0.4
    alpha.delta = 1e-05
    problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-06)
    self.assertAlmostEqual(x.value, 1 - 0.4 ** 2 - 0.4 ** 3)
    problem.backward()
    problem.derivative()
    self.assertAlmostEqual(alpha.gradient, -2 * 0.4 - 3 * 0.4 ** 2)
    self.assertAlmostEqual(x.delta, alpha.gradient * 1e-05)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)