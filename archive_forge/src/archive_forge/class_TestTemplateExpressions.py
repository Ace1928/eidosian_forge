import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
class TestTemplateExpressions(unittest.TestCase):

    def setUp(self):
        self.m = m = ConcreteModel()
        m.I = RangeSet(1, 9)
        m.J = RangeSet(10, 19)
        m.x = Var(m.I, initialize=lambda m, i: i + 1)
        m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
        m.p = Param(m.I, m.J, initialize=lambda m, i, j: 100 * i + j)
        m.s = Set(m.I, initialize=lambda m, i: range(i))

    def test_nonTemplates(self):
        m = self.m
        self.assertIs(resolve_template(m.x[1]), m.x[1])
        e = m.x[1] + m.x[2]
        self.assertIs(resolve_template(e), e)

    def test_IndexTemplate(self):
        m = self.m
        i = IndexTemplate(m.I)
        with self.assertRaisesRegex(TemplateExpressionError, 'Evaluating uninitialized IndexTemplate'):
            value(i)
        self.assertEqual(str(i), '{I}')
        i.set_value(5)
        self.assertEqual(value(i), 5)
        self.assertIs(resolve_template(i), 5)

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

    def test_template_scalar_with_set(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.s[t]
        self.assertIs(type(e), EXPR.NPV_Structural_GetItemExpression)
        self.assertEqual(e.args, (m.s, t))
        self.assertFalse(e.is_constant())
        self.assertTrue(e.is_fixed())
        ee = e.polynomial_degree()
        self.assertIs(type(ee), EXPR.CallExpression)
        t.set_value(1)
        with self.assertRaisesRegex(AttributeError, "'_InsertionOrderSetData' object has no attribute 'polynomial_degree'"):
            e.polynomial_degree()
        self.assertEqual(str(e), 's[{I}]')
        t.set_value(5)
        v = e()
        self.assertIs(v, m.s[5])
        self.assertIs(resolve_template(e), m.s[5])
        t.set_value()

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

    def test_nested_template_operation(self):
        m = self.m
        t = IndexTemplate(m.I)
        e = m.x[t + m.P[t + 1]]
        self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
        self.assertEqual(str(e), 'x[{I} + P[{I} + 1]]')

    def test_block_templates(self):
        m = ConcreteModel()
        m.T = RangeSet(3)

        @m.Block(m.T)
        def b(b, i):
            b.x = Var(initialize=i)

            @b.Block(m.T)
            def bb(bb, j):
                bb.I = RangeSet(i * j)
                bb.y = Var(bb.I, initialize=lambda m, i: i)
        t = IndexTemplate(m.T)
        e = m.b[t].x
        self.assertIs(type(e), EXPR.Numeric_GetAttrExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertIs(type(e.arg(0)), EXPR.NPV_Structural_GetItemExpression)
        self.assertIs(e.arg(0).arg(0), m.b)
        self.assertEqual(e.arg(0).nargs(), 2)
        self.assertIs(e.arg(0).arg(1), t)
        self.assertEqual(str(e), 'b[{T}].x')
        t.set_value(2)
        v = e()
        self.assertIn(type(v), (int, float))
        self.assertEqual(v, 2)
        self.assertIs(resolve_template(e), m.b[2].x)
        t.set_value()
        e = m.b[t].bb[t].y[1]
        self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
        self.assertEqual(e.nargs(), 2)
        self.assertEqual(str(e), 'b[{T}].bb[{T}].y[1]')
        t.set_value(2)
        v = e()
        self.assertIn(type(v), (int, float))
        self.assertEqual(v, 1)
        self.assertIs(resolve_template(e), m.b[2].bb[2].y[1])

    def test_template_name(self):
        m = self.m
        t = IndexTemplate(m.I)
        E = m.x[t + m.P[1 + t]] + m.P[1]
        self.assertEqual(str(E), 'x[{I} + P[1 + {I}]] + P[1]')
        E = m.x[t + m.P[1 + t] ** 2.0] ** 2.0 + m.P[1]
        self.assertEqual(str(E), 'x[{I} + P[1 + {I}]**2.0]**2.0 + P[1]')

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

    def test_clone(self):
        m = self.m
        t = IndexTemplate(m.I)
        E_base = m.x[t + m.P[t + 1]] + m.P[1]
        E = E_base.clone()
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(0)
        self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)), type(E_base.arg(0).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1), E_base.arg(0).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
        E_base = m.P[1] + m.x[t + m.P[t + 1]]
        E = E_base.clone()
        self.assertTrue(isinstance(E, EXPR.SumExpressionBase))
        e = E.arg(1)
        self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)), type(E_base.arg(1).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1), E_base.arg(1).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
        E_base = m.x[t + m.P[t + 1]] + 1
        E = E_base.clone()
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(0)
        self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)), type(E_base.arg(0).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1), E_base.arg(0).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)
        E_base = 1 + m.x[t + m.P[t + 1]]
        E = E_base.clone()
        self.assertIsInstance(E, EXPR.SumExpressionBase)
        e = E.arg(-1)
        self.assertIs(type(e), EXPR.Numeric_GetItemExpression)
        self.assertIsNot(e, E_base.arg(0))
        self.assertEqual(e.nargs(), 2)
        self.assertIs(e.arg(0), m.x)
        self.assertIsInstance(e.arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(0), t)
        self.assertIs(type(e.arg(1).arg(1)), EXPR.NPV_Numeric_GetItemExpression)
        self.assertIs(type(e.arg(1).arg(1)), type(E_base.arg(-1).arg(1).arg(1)))
        self.assertIsNot(e.arg(1).arg(1), E_base.arg(-1).arg(1).arg(1))
        self.assertIsInstance(e.arg(1).arg(1).arg(1), EXPR.SumExpressionBase)
        self.assertIs(e.arg(1).arg(1).arg(1).arg(0), t)