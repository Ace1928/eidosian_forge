import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
def test_single_derivative_list(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    e = differentiate(1, wrt_list=[m.x])
    self.assertIs(type(e), list)
    self.assertEqual(len(e), 1)
    e = e[0]
    self.assertIn(type(e), (int, float))
    self.assertEqual(e, 0)
    e = differentiate(m.x, wrt_list=[m.x])
    self.assertIs(type(e), list)
    self.assertEqual(len(e), 1)
    e = e[0]
    self.assertIn(type(e), (int, float))
    self.assertEqual(e, 1)
    e = differentiate(m.x ** 2, wrt_list=[m.x])
    self.assertIs(type(e), list)
    self.assertEqual(len(e), 1)
    e = e[0]
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(2.0 * m.x))
    e = differentiate(m.y, wrt_list=[m.x])
    self.assertIs(type(e), list)
    self.assertEqual(len(e), 1)
    e = e[0]
    self.assertIn(type(e), (int, float))
    self.assertEqual(e, 0)
    e = differentiate(m.x * m.y, wrt_list=[m.x])
    self.assertIs(type(e), list)
    self.assertEqual(len(e), 1)
    e = e[0]
    self.assertIs(e, m.y)
    self.assertEqual(s(e), s(m.y))
    e = differentiate(m.x ** 2 * m.y, wrt_list=[m.x])
    self.assertIs(type(e), list)
    self.assertEqual(len(e), 1)
    e = e[0]
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(2.0 * m.x * m.y))
    e = differentiate(m.x ** 2 / m.y, wrt_list=[m.x])
    self.assertIs(type(e), list)
    self.assertEqual(len(e), 1)
    e = e[0]
    self.assertTrue(e.is_expression_type())
    self.assertEqual(s(e), s(2.0 * m.x * m.y ** (-1.0)))