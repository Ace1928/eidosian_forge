import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_sum_pulses_in_place(self):
    m = self.get_model()
    p1 = Pulse(interval_var=m.a, height=2)
    expr = p1
    self.assertIsInstance(expr, Pulse)
    self.assertEqual(len(expr.args), 2)
    self.assertEqual(expr.nargs(), 2)
    p2 = Pulse(interval_var=m.b, height=3)
    expr += p2
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(len(expr.args), 2)
    self.assertEqual(expr.nargs(), 2)
    self.assertIs(expr.args[0], p1)
    self.assertIs(expr.args[1], p2)