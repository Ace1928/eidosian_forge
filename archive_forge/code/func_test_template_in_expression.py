import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_template_in_expression(self):
    m = self.m
    t = IndexTemplate(m.I)
    E = m.x[t + m.P[t + 1]] + m.P[1]
    self.assertIsInstance(E, EXPR.SumExpressionBase)
    e = E.arg(0)
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.x)
    self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(0), t)
    self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
    self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
    E = m.P[1] + m.x[t + m.P[t + 1]]
    self.assertIsInstance(E, EXPR.SumExpressionBase)
    e = E.arg(1)
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.x)
    self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(0), t)
    self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
    self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
    E = m.x[t + m.P[t + 1]] + 1
    self.assertIsInstance(E, EXPR.SumExpressionBase)
    e = E.arg(0)
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.x)
    self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(0), t)
    self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
    self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
    E = 1 + m.x[t + m.P[t + 1]]
    self.assertIsInstance(E, EXPR.SumExpressionBase)
    e = E.arg(E.nargs() - 1)
    self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
    self.assertEqual(e.nargs(), 2)
    self.assertIs(e.arg(0), m.x)
    self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(0), t)
    self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
    self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
    self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)