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
def test_sum_largest(self) -> None:
    """Test the sum_largest atom and related atoms.
        """
    with self.assertRaises(Exception) as cm:
        cp.sum_largest(self.x, -1)
    self.assertEqual(str(cm.exception), 'Second argument must be a positive integer.')
    with self.assertRaises(Exception) as cm:
        cp.lambda_sum_largest(self.x, 2.4)
    self.assertEqual(str(cm.exception), 'First argument must be a square matrix.')
    with self.assertRaises(Exception) as cm:
        cp.lambda_sum_largest(Variable((2, 2)), 2.4)
    self.assertEqual(str(cm.exception), 'Second argument must be a positive integer.')
    with self.assertRaises(ValueError) as cm:
        cp.lambda_sum_largest([[1, 2], [3, 4]], 2).value
    self.assertEqual(str(cm.exception), 'Input matrix was not Hermitian/symmetric.')
    atom = cp.sum_largest(self.x, 2)
    copy = atom.copy()
    self.assertTrue(type(copy) is type(atom))
    self.assertEqual(copy.args, atom.args)
    self.assertFalse(copy.args is atom.args)
    self.assertEqual(copy.get_data(), atom.get_data())
    copy = atom.copy(args=[self.y])
    self.assertTrue(type(copy) is type(atom))
    self.assertTrue(copy.args[0] is self.y)
    self.assertEqual(copy.get_data(), atom.get_data())
    atom = cp.lambda_sum_largest(Variable((2, 2)), 2)
    copy = atom.copy()
    self.assertTrue(type(copy) is type(atom))
    atom = cp.sum_largest(self.x, 2)
    assert atom.is_pwl()