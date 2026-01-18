import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_template_operation(self):
    m = self.m
    t = IndexTemplate(m.I)
    e = m.x[t + m.P[5]]
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.x)
    self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(0), t)
    self.assertIs(e.arg(1).arg(1), m.P[5])
    self.assertEqual(str(e), 'x[{I} + P[5]]')