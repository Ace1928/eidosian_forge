import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_subtract_two_steps(self):
    m = self.get_model()
    s = Step(m.a.start_time, height=2) - Step(m.b.start_time, height=5)
    self.assertIsInstance(s, CumulativeFunction)
    self.assertEqual(len(s.args), 2)
    self.assertEqual(s.nargs(), 2)
    self.assertIsInstance(s.args[0], StepAtStart)
    self.assertIsInstance(s.args[1], NegatedStepFunction)
    self.assertEqual(len(s.args[1].args), 1)
    self.assertEqual(s.args[1].nargs(), 1)
    self.assertIsInstance(s.args[1].args[0], StepAtStart)