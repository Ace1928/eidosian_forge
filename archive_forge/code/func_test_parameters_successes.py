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
def test_parameters_successes(self) -> None:
    p = Parameter(name='p')
    self.assertEqual(p.name(), 'p')
    self.assertEqual(p.shape, tuple())
    val = -np.ones((4, 3))
    val[0, 0] = 2
    p = Parameter((4, 3))
    p.value = val
    p = Parameter(value=10)
    self.assertEqual(p.value, 10)
    p.value = 10
    p.value = None
    self.assertEqual(p.value, None)
    p = Parameter((4, 3), nonpos=True)
    self.assertEqual(repr(p), 'Parameter((4, 3), nonpos=True)')
    p = Parameter((2, 2), diag=True)
    p.value = sp.csc_matrix(np.eye(2))
    self.assertItemsAlmostEqual(p.value.todense(), np.eye(2), places=10)