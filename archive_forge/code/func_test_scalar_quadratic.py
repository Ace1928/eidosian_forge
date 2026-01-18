import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
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