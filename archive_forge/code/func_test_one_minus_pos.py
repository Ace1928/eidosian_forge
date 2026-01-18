import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_one_minus_pos(self) -> None:
    x = cp.Variable(pos=True)
    a = cp.Parameter(pos=True, value=3)
    b = cp.Parameter(pos=True, value=0.1)
    obj = cp.Maximize(x)
    constr = [cp.one_minus_pos(a * x) >= a * b]
    problem = cp.Problem(obj, constr)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)
    perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)