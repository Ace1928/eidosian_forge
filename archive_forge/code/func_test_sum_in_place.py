import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_sum_in_place(self):
    m = self.get_model()
    expr = Step(m.a.start_time, height=4) + Pulse(interval_var=m.b, height=-1)
    expr += Step(0, 1)
    self.assertEqual(len(expr.args), 3)
    self.assertEqual(expr.nargs(), 3)
    self.assertIsInstance(expr.args[0], StepAtStart)
    self.assertIsInstance(expr.args[1], Pulse)
    self.assertIsInstance(expr.args[2], StepAt)
    self.assertEqual(str(expr), 'Step(a.start_time, height=4) + Pulse(b, height=-1) + Step(0, height=1)')