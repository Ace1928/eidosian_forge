import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_rule_immutable_param_expr(self):
    """Test rule option that returns a single immutable param for the expression"""
    model = self.create_model()

    def f(model, i):
        return model.p[i]
    model.p = Param(RangeSet(1, 4), initialize=1.0, mutable=False)
    model.x = Var()
    model.obj = Objective(model.A, rule=f)
    self.assertEqual(model.obj[2](), 1.0)
    self.assertEqual(value(model.obj[2]), 1.0)