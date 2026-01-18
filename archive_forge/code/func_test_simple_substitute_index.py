import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_simple_substitute_index(self):

    def diffeq(m, t, i):
        return m.dxdt[t, i] == t * m.x[t, i] ** 2 + m.y ** 2
    m = self.m
    t = IndexTemplate(m.TIME)
    e = diffeq(m, t, 2)
    t.set_value(5)
    self.assertTrue(isinstance(e, EXPR.RelationalExpression))
    self.assertEqual((e.arg(0)(), e.arg(1)()), (10, 126))
    E = substitute_template_expression(e, substitute_template_with_value)
    self.assertIsNot(e, E)
    self.assertEqual(str(E), 'dxdt[5,2]  ==  5.0*x[5,2]**2 + y**2')