import sys
import unittest
from typing import Tuple
import numpy
import scipy.sparse as sp
import cvxpy.interface.matrix_utilities as intf
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
class TestMatrices(unittest.TestCase):
    """ Unit tests for testing different forms of matrices as constants. """

    def assertExpression(self, expr, shape: Tuple[int, ...]) -> None:
        """Asserts that expr is an Expression with dimension shape.
        """
        assert isinstance(expr, Expression) or isinstance(expr, Constraint)
        self.assertEqual(expr.shape, shape)

    def setUp(self) -> None:
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')
        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')
        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_numpy_arrays(self) -> None:
        v = numpy.arange(2)
        self.assertExpression(self.x + v, (2,))
        self.assertExpression(v + self.x, (2,))
        self.assertExpression(self.x - v, (2,))
        self.assertExpression(v - self.x, (2,))
        self.assertExpression(self.x <= v, (2,))
        self.assertExpression(v <= self.x, (2,))
        self.assertExpression(self.x == v, (2,))
        self.assertExpression(v == self.x, (2,))
        A = numpy.arange(8).reshape((4, 2))
        self.assertExpression(A @ self.x, (4,))
        A = numpy.ones((2, 2))
        self.assertExpression(A << self.A, (2, 2))
        self.assertExpression(A >> self.A, (2, 2))

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

    def test_numpy_scalars(self) -> None:
        """Test numpy scalars."""
        v = numpy.float64(2.0)
        self.assertExpression(self.x + v, (2,))
        self.assertExpression(v + self.x, (2,))
        self.assertExpression(v * self.x, (2,))
        self.assertExpression(self.x - v, (2,))
        self.assertExpression(v - v - self.x, (2,))
        self.assertExpression(self.x <= v, (2,))
        self.assertExpression(v <= self.x, (2,))
        self.assertExpression(self.x == v, (2,))
        self.assertExpression(v == self.x, (2,))
        self.assertExpression(v << self.A, (2, 2))
        self.assertExpression(v >> self.A, (2, 2))

    def test_scipy_sparse(self) -> None:
        """Test scipy sparse matrices."""
        A = numpy.arange(8).reshape((4, 2))
        A = sp.csc_matrix(A)
        A = sp.eye(2).tocsc()
        key = (slice(0, 1, None), slice(None, None, None))
        Aidx = intf.index(A, (slice(0, 2, None), slice(None, None, None)))
        Aidx = intf.index(Aidx, key)
        self.assertEqual(Aidx.shape, (1, 2))
        self.assertEqual(Aidx[0, 0], 1)
        self.assertEqual(Aidx[0, 1], 0)
        var = Variable((4, 2))
        A = numpy.arange(8).reshape((4, 2))
        A = sp.csc_matrix(A)
        B = sp.hstack([A, A])
        self.assertExpression(var + A, (4, 2))
        self.assertExpression(A + var, (4, 2))
        self.assertExpression(B @ var, (4, 2))
        self.assertExpression(var - A, (4, 2))
        self.assertExpression(A - A - var, (4, 2))
        if PY35:
            self.assertExpression(var.__rmatmul__(B), (4, 2))