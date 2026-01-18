import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_paper_example_exp_log(self) -> None:
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    a = cp.Parameter(pos=True, value=0.2)
    b = cp.Parameter(pos=True, value=0.3)
    obj = cp.Minimize(x * y)
    constr = [cp.exp(a * y / x) <= cp.log(b * y)]
    problem = cp.Problem(obj, constr)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.01, max_iters=10000)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.01, max_iters=5000)