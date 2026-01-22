import os
import unittest
import numpy as np
import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.base_test import BaseTest
@unittest.skipUnless('GUROBI' in INSTALLED_SOLVERS, 'GUROBI is not installed.')
class TestGurobiWrite(BaseTest):

    def test_write(self) -> None:
        """Test the Gurobi model.write().
        """
        if not os.path.exists('./resources/'):
            os.makedirs('./resources/')
        elif os.path.exists('./resources/gurobi_model.lp'):
            os.remove('./resources/gurobi_model.lp')
        m = 20
        n = 15
        np.random.seed(0)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x = cp.Variable(n)
        cost = cp.sum_squares(A @ x - b)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve(solver=cp.GUROBI, verbose=True, save_file='./resources/gurobi_model.lp')
        assert os.path.exists('./resources/gurobi_model.lp')