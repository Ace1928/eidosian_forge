import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_rule_var_expr(self):
    """Test rule option that returns a single var for the expression"""
    model = self.create_model()

    def f(model, i):
        return model.x[i]
    model.r = RangeSet(1, 4)
    model.x = Var(model.r, initialize=1.0)
    model.obj = Objective(model.A, rule=f)
    self.assertEqual(model.obj[2](), 1.0)
    self.assertEqual(value(model.obj[2]), 1.0)