import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar
from pyomo.contrib.cp.scheduling_expr.precedence_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_start_after_end(self):
    m = self.get_model()
    m.c = LogicalConstraint(expr=m.a.start_time.after(m.b.end_time, delay=2))
    self.assertIsInstance(m.c.expr, BeforeExpression)
    self.assertEqual(len(m.c.expr.args), 3)
    self.assertIs(m.c.expr.args[0], m.b.end_time)
    self.assertIs(m.c.expr.args[1], m.a.start_time)
    self.assertEqual(m.c.expr.delay, 2)
    self.assertEqual(str(m.c.expr), 'b.end_time + 2 <= a.start_time')