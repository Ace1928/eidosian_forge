import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_special_idx(self) -> None:
    """Test with special index.
        """
    c = [0, 1]
    n = len(c)
    f = cp.Variable((n, n), hermitian=True)
    constraints = [f >> 0]
    for k in range(1, n):
        indices = [i * n + i - (n - k) for i in range(n - k, n)]
        constraints += [cp.sum(cp.vec(f)[indices]) == c[n - k]]
    obj = cp.Maximize(c[0] - cp.real(cp.trace(f)))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver='SCS')