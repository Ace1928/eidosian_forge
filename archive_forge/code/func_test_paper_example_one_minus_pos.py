import warnings
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
def test_paper_example_one_minus_pos(self) -> None:
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    a = cp.Parameter(pos=True, value=2)
    b = cp.Parameter(pos=True, value=1)
    c = cp.Parameter(pos=True, value=3)
    obj = cp.Minimize(x * y)
    constr = [(y * cp.one_minus_pos(x / y)) ** a >= b, x >= y / c]
    problem = cp.Problem(obj, constr)
    gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=0.001)
    perturbcheck(problem, solve_methods=[s.SCS], gp=True, atol=0.001)