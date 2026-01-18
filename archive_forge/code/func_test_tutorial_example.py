import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dqcp2dcp.dqcp2dcp import Dqcp2Dcp
from cvxpy.reductions.solvers import bisection
from cvxpy.tests import base_test
def test_tutorial_example(self) -> None:
    x = cp.Variable()
    y = cp.Variable(pos=True)
    objective_fn = -cp.sqrt(x) / y
    problem = cp.Problem(cp.Minimize(objective_fn), [cp.exp(x) <= y])
    problem.solve(SOLVER, qcp=True)