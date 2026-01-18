import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_template_scalar(self):
    m = self.m
    t = IndexTemplate(m.I)
    e = m.x[t]
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.args, (m.x, t))
    self.assertFalse(e.is_constant())
    self.assertFalse(e.is_fixed())
    self.assertEqual(e.polynomial_degree(), 1)
    self.assertEqual(str(e), 'x[{I}]')
    t.set_value(5)
    v = e()
    self.assertIn(type(v), (int, float))
    self.assertEqual(v, 6)
    self.assertIs(resolve_template(e), m.x[5])
    t.set_value()
    e = m.p[t, 10]
    self.assertIs(type(e), EXPR.NPV_Numeric_GetItemExpression)
    self.assertEqual(e.args, (m.p, t, 10))
    self.assertFalse(e.is_constant())
    self.assertTrue(e.is_fixed())
    self.assertEqual(e.polynomial_degree(), 0)
    self.assertEqual(str(e), 'p[{I},10]')
    t.set_value(5)
    v = e()
    self.assertIn(type(v), (int, float))
    self.assertEqual(v, 510)
    self.assertIs(resolve_template(e), m.p[5, 10])
    t.set_value()
    e = m.p[5, t]
    self.assertIs(type(e), EXPR.NPV_Numeric_GetItemExpression)
    self.assertEqual(e.args, (m.p, 5, t))
    self.assertFalse(e.is_constant())
    self.assertTrue(e.is_fixed())
    self.assertEqual(e.polynomial_degree(), 0)
    self.assertEqual(str(e), 'p[5,{I}]')
    t.set_value(10)
    v = e()
    self.assertIn(type(v), (int, float))
    self.assertEqual(v, 510)
    self.assertIs(resolve_template(e), m.p[5, 10])
    t.set_value()