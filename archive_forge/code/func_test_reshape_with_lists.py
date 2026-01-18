import unittest
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable
from cvxpy.utilities import shape
def test_reshape_with_lists(self) -> None:
    n = 2
    a = Variable([n, n])
    b = Variable(n ** 2)
    c = reshape(b, [n, n])
    self.assertEqual((a + c).shape, (n, n))
    d = reshape(b, (n, n))
    self.assertEqual((a + d).shape, (n, n))