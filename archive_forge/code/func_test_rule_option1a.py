import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_rule_option1a(self):
    """Test rule option"""
    model = self.create_model()
    model.B = RangeSet(1, 4)

    @simple_objectivelist_rule
    def f(model, i):
        if i > 4:
            return None
        ans = 0
        for j in model.B:
            ans = ans + model.x[j]
        ans *= i
        return ans
    model.x = Var(model.B, initialize=2)
    model.o = ObjectiveList(rule=f)
    self.assertEqual(model.o[1](), 8)
    self.assertEqual(model.o[2](), 16)
    self.assertEqual(len(model.o), 4)