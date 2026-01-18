import pyomo.common.unittest as unittest
from pyomo.contrib.cp.interval_var import (
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var
def test_optional_rule(self):
    m = ConcreteModel()
    m.idx = Set(initialize=[(4, 2), (5, 2)], dimen=2)

    def optional_rule(m, i, j):
        return i % j == 0
    m.act = IntervalVar(m.idx, optional=optional_rule)
    self.assertTrue(m.act[4, 2].optional)
    self.assertFalse(m.act[5, 2].optional)