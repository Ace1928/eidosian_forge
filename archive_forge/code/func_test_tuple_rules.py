import pyomo.common.unittest as unittest
from pyomo.environ import (
import pyomo.core.expr as EXPR
from pyomo.core.expr.template_expr import (
def test_tuple_rules(self):
    m = ConcreteModel()
    m.I = RangeSet(3)
    m.x = Var(m.I)

    @m.Constraint(m.I)
    def c(m, i):
        return (None, m.x[i], 0)
    template, indices = templatize_constraint(m.c)
    self.assertEqual(len(indices), 1)
    self.assertIs(indices[0]._set, m.I)
    self.assertEqual(str(template), 'x[_1]  <=  0')
    self.assertEqual(list(m.I), list(range(1, 4)))
    indices[0].set_value(2)
    self.assertEqual(str(resolve_template(template)), 'x[2]  <=  0')

    @m.Constraint(m.I)
    def d(m, i):
        return (0, m.x[i], 10)
    template, indices = templatize_constraint(m.d)
    self.assertEqual(len(indices), 1)
    self.assertIs(indices[0]._set, m.I)
    self.assertEqual(str(template), '0  <=  x[_1]  <=  10')
    self.assertEqual(list(m.I), list(range(1, 4)))
    indices[0].set_value(2)
    self.assertEqual(str(resolve_template(template)), '0  <=  x[2]  <=  10')

    @m.Constraint(m.I)
    def e(m, i):
        return (m.x[i], 0)
    template, indices = templatize_constraint(m.e)
    self.assertEqual(len(indices), 1)
    self.assertIs(indices[0]._set, m.I)
    self.assertEqual(str(template), 'x[_1]  ==  0')
    self.assertEqual(list(m.I), list(range(1, 4)))
    indices[0].set_value(2)
    self.assertEqual(str(resolve_template(template)), 'x[2]  ==  0')