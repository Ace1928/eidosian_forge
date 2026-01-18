import pyomo.common.unittest as unittest
from pyomo.contrib.cp import IntervalVar, Step, Pulse
from pyomo.contrib.cp.scheduling_expr.step_function_expressions import (
from pyomo.environ import ConcreteModel, LogicalConstraint
def test_sum_step_and_cumul_func(self):
    m = self.get_model()
    s1 = Step(m.a.start_time, height=4)
    p1 = Step(m.a.start_time, height=4)
    cumul = s1 + p1
    s = Step(m.a.end_time, height=3)
    expr = s + cumul
    self.assertIsInstance(expr, CumulativeFunction)
    self.assertEqual(expr.nargs(), 3)
    self.assertIs(expr.args[0], s)
    self.assertIs(expr.args[1], s1)
    self.assertIs(expr.args[2], p1)