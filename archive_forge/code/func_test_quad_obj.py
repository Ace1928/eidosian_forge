import math
import unittest
import numpy as np
import pytest
import scipy.linalg as la
import scipy.stats as st
import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
from cvxpy.utilities.versioning import Version
def test_quad_obj(self) -> None:
    """Test SCS canonicalization with a quadratic objective.
        """
    import scs
    if Version(scs.__version__) >= Version('3.0.0'):
        x = cp.Variable(2)
        expr = cp.sum_squares(x)
        constr = [x >= 1]
        prob = cp.Problem(cp.Minimize(expr), constr)
        data = prob.get_problem_data(solver=cp.SCS)
        self.assertItemsAlmostEqual(data[0]['P'].A, 2 * np.eye(2))
        solution1 = prob.solve(solver=cp.SCS)
        prob = cp.Problem(cp.Minimize(expr), constr)
        solver_opts = {'use_quad_obj': False}
        data = prob.get_problem_data(solver=cp.SCS, solver_opts=solver_opts)
        assert 'P' not in data[0]
        solution2 = prob.solve(solver=cp.SCS, **solver_opts)
        assert np.isclose(solution1, solution2)
        expr = cp.norm(x, 1)
        prob = cp.Problem(cp.Minimize(expr), constr)
        data = prob.get_problem_data(solver=cp.SCS)
        assert 'P' not in data[0]