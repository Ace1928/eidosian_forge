import warnings
import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import (
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.linalg import gershgorin_psd_check
def test_none_idx(self) -> None:
    """Test None as index.
        """
    expr = self.a[None, None]
    self.assertEqual(expr.shape, (1, 1))
    expr = self.x[:, None]
    self.assertEqual(expr.shape, (2, 1))
    expr = self.x[None, :]
    self.assertEqual(expr.shape, (1, 2))
    expr = Constant([1, 2])[None, :]
    self.assertEqual(expr.shape, (1, 2))
    self.assertItemsAlmostEqual(expr.value, [1, 2])