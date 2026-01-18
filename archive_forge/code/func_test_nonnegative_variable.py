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
def test_nonnegative_variable(self) -> None:
    """Test the NonNegative Variable class.
        """
    x = Variable(nonneg=True)
    p = Problem(Minimize(5 + x), [x >= 3])
    p.solve(solver=cp.SCS, eps=1e-05)
    self.assertAlmostEqual(p.value, 8)
    self.assertAlmostEqual(x.value, 3)