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
def test_partial_optimize_min_1norm(self) -> None:
    dims = 3
    x, t = (Variable(dims), Variable(dims))
    p1 = Problem(Minimize(cp.sum(t)), [-t <= x, x <= t])
    g = partial_optimize(p1, [t], [x], solver='ECOS')
    p2 = Problem(Minimize(g))
    p2.solve(solver='ECOS')
    p1.solve(solver='ECOS')
    self.assertAlmostEqual(p1.value, p2.value)