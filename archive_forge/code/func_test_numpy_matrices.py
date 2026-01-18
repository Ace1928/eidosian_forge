import sys
import unittest
from typing import Tuple
import numpy
import scipy.sparse as sp
import cvxpy.interface.matrix_utilities as intf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
def test_numpy_matrices(self) -> None:
    v = numpy.arange(2)
    self.assertExpression(self.x + v, (2,))
    self.assertExpression(v + v + self.x, (2,))
    self.assertExpression(self.x - v, (2,))
    self.assertExpression(v - v - self.x, (2,))
    self.assertExpression(self.x <= v, (2,))
    self.assertExpression(v <= self.x, (2,))
    self.assertExpression(self.x == v, (2,))
    self.assertExpression(v == self.x, (2,))
    A = numpy.arange(8).reshape((4, 2))
    self.assertExpression(A @ self.x, (4,))
    self.assertExpression(A.T.dot(A) @ self.x, (2,))
    A = numpy.ones((2, 2))
    self.assertExpression(A << self.A, (2, 2))
    self.assertExpression(A >> self.A, (2, 2))