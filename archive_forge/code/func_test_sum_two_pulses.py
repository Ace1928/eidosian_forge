import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_sum_two_pulses(self):
    m = self.get_model()
    m.p1 = Pulse(interval_var=m.a, height=3)
    m.p2 = Pulse(interval_var=m.b, height=-2)
    expr = m.p1 + m.p2
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(len(expr.args), 2)
    self.assertEqual(expr.nargs(), 2)
    self.assertIs(expr.args[0], m.p1)
    self.assertIs(expr.args[1], m.p2)