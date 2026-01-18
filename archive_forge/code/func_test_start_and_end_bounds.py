import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_start_and_end_bounds(self):
    m = ConcreteModel()
    m.i = IntervalVar(start=(0, 5))
    self.assertEqual(m.i.start_time.lower, 0)
    self.assertEqual(m.i.start_time.upper, 5)
    m.i.end_time.bounds = (12, 14)
    self.assertEqual(m.i.end_time.lower, 12)
    self.assertEqual(m.i.end_time.upper, 14)