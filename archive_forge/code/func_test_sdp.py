import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
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