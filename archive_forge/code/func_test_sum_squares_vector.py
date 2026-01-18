import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_sum_squares_vector(self) -> None:
    alpha = cp.Parameter(shape=(2,), pos=True, value=[1.0, 1.0])
    beta = cp.Parameter(pos=True, value=20)
    kappa = cp.Parameter(pos=True, value=10)
    w = cp.Variable(2, pos=True)
    h = cp.Variable(2, pos=True)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(alpha + h)), [cp.multiply(w, h) >= beta, cp.sum(alpha + w) <= kappa])
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.1, max_iters=1000)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.1, max_iters=1000)