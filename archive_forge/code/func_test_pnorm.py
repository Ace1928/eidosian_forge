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
def test_pnorm(self) -> None:
    atom = cp.pnorm(self.x, p=1.5)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=1)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=2)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    self.assertEqual(atom.sign, s.NONNEG)
    expr = cp.norm(self.A, 2, axis=0)
    self.assertEqual(expr.shape, (2,))
    expr = cp.norm(self.A, 2, axis=0, keepdims=True)
    self.assertEqual(expr.shape, (1, 2))
    expr = cp.norm(self.A, 2, axis=1, keepdims=True)
    self.assertEqual(expr.shape, (2, 1))
    atom = cp.pnorm(self.x, p='inf')
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p='Inf')
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=np.inf)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONVEX)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=0.5)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONCAVE)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=0.7)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONCAVE)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=-0.1)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONCAVE)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=-1)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONCAVE)
    self.assertEqual(atom.sign, s.NONNEG)
    atom = cp.pnorm(self.x, p=-1.3)
    self.assertEqual(atom.shape, tuple())
    self.assertEqual(atom.curvature, s.CONCAVE)
    self.assertEqual(atom.sign, s.NONNEG)
    copy = atom.copy()
    self.assertTrue(type(copy) is type(atom))
    self.assertEqual(copy.args, atom.args)
    self.assertFalse(copy.args is atom.args)
    self.assertEqual(copy.get_data(), atom.get_data())
    copy = atom.copy(args=[self.y])
    self.assertTrue(type(copy) is type(atom))
    self.assertTrue(copy.args[0] is self.y)
    self.assertEqual(copy.get_data(), atom.get_data())