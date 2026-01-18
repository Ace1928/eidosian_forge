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
@unittest.skipUnless(len(INSTALLED_MI_SOLVERS) > 0, 'No mixed-integer solver is installed.')
def test_partial_optimize_special_var(self) -> None:
    x, y = (Variable(boolean=True), Variable(integer=True))
    p1 = Problem(Minimize(x + y), [x + y >= 3, y >= 4, x >= 5])
    p1.solve(solver=cp.ECOS_BB)
    p2 = Problem(Minimize(y), [x + y >= 3, y >= 4])
    g = partial_optimize(p2, [y], [x])
    p3 = Problem(Minimize(x + g), [x >= 5])
    p3.solve(solver=cp.ECOS_BB)
    self.assertAlmostEqual(p1.value, p3.value)