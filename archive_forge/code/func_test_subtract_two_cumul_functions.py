import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_subtract_two_cumul_functions(self):
    m = self.get_model()
    p1 = Pulse(interval_var=m.a, height=2)
    s1 = Step(m.a.start_time, height=4)
    p2 = Pulse(interval_var=m.b, height=3)
    p3 = Pulse(interval_var=m.a, height=-4)
    cumul1 = s1 - p2
    cumul2 = p2 + p3
    expr = cumul1 - cumul2
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(expr.nargs(), 4)
    self.assertIs(expr.args[0], s1)
    self.assertIsInstance(expr.args[1], NegatedStepFunction)
    self.assertIs(expr.args[1].args[0], p2)
    self.assertIsInstance(expr.args[2], NegatedStepFunction)
    self.assertIs(expr.args[2].args[0], p2)
    self.assertIsInstance(expr.args[3], NegatedStepFunction)
    self.assertIs(expr.args[3].args[0], p3)