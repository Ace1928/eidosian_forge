import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_rule_option(self):
    """Test rule option"""
    model = ConcreteModel()

    def f(model):
        ans = 0
        for i in [1, 2, 3, 4]:
            ans = ans + model.x[i]
        return ans
    model.x = Var(RangeSet(1, 4), initialize=2)
    model.obj = Objective(rule=f)
    self.assertEqual(model.obj(), 8)
    self.assertEqual(value(model.obj), 8)
    self.assertEqual(value(model.obj._data[None]), 8)