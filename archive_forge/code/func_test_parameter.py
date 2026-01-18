import unittest
import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA
import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize
def test_parameter(self):
    x_val = np.array([1, 3, 2, -5, 0])
    assert cp.dotsort(self.x, cp.Parameter(2, pos=True)).is_incr(0)
    assert cp.dotsort(self.x, cp.Parameter(2, nonneg=True)).is_incr(0)
    assert not cp.dotsort(self.x, cp.Parameter(2, neg=True)).is_incr(0)
    assert cp.dotsort(self.x, cp.Parameter(2, neg=True)).is_decr(0)
    w_p = cp.Parameter(2, value=[1, 0])
    expr = cp.dotsort(self.x, w_p)
    assert not expr.is_incr(0)
    assert not expr.is_decr(0)
    prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
    prob.solve(enforce_dpp=True)
    self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([1, 0, 0, 0, 0])))
    w_p.value = [-1, -1]
    prob.solve(enforce_dpp=True)
    self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([-1, -1, 0, 0, 0])))
    w_p = cp.Parameter(2, value=[1, 0])
    parameter_affine_expression = 2 * w_p
    expr = cp.dotsort(self.x, parameter_affine_expression)
    prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
    prob.solve(enforce_dpp=True)
    self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([2, 0, 0, 0, 0])))
    w_p.value = [-1, -1]
    prob.solve(enforce_dpp=True)
    self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(np.array([-2, -2, 0, 0, 0])))
    x_const = np.array([1, 2, 3])
    p = cp.Parameter(value=2)
    p_squared = p ** 2
    expr = cp.dotsort(x_const, p_squared)
    problem = cp.Problem(cp.Minimize(expr))
    problem.solve(enforce_dpp=True)
    self.assertAlmostEqual(expr.value, 2 ** 2 * 3)
    p.value = -1
    problem.solve(enforce_dpp=True)
    self.assertAlmostEqual(expr.value, (-1) ** 2 * 3)
    with pytest.warns(UserWarning, match='You are solving a parameterized problem that is not DPP.'):
        x_val = np.array([1, 2, 3, 4, 5])
        p = cp.Parameter(value=2)
        p_squared = p ** 2
        expr = cp.dotsort(self.x, p_squared)
        problem = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        problem.solve()
        self.assertAlmostEqual(expr.value, 2 ** 2 * 5)
        p.value = -1
        problem.solve()
        self.assertAlmostEqual(expr.value, (-1) ** 2 * 5)