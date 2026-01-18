import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_rank_one_nmf(self) -> None:
    X = cp.Variable((3, 3), pos=True)
    x = cp.Variable((3,), pos=True)
    y = cp.Variable((3,), pos=True)
    xy = cp.vstack([x[0] * y, x[1] * y, x[2] * y])
    a = cp.Parameter(value=-1.0)
    b = cp.Parameter(pos=True, shape=(6,), value=np.array([1.0, 1.9, 0.8, 3.2, 5.9, 1.0]))
    R = cp.maximum(cp.multiply(X, xy ** a), cp.multiply(X ** a, xy))
    objective = cp.sum(R)
    constraints = [X[0, 0] == b[0], X[0, 2] == b[1], X[1, 1] == b[2], X[2, 0] == b[3], X[2, 1] == b[4], x[0] * x[1] * x[2] == b[5]]
    problem = cp.Problem(cp.Minimize(objective), constraints)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.01, max_iters=1000)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.01, max_iters=1000)