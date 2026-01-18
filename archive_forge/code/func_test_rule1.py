import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_rule1(self):
    """Test rule option"""
    model = ConcreteModel()
    model.B = RangeSet(1, 4)

    def f(model):
        ans = 0
        for i in model.B:
            ans = ans + model.x[i]
        ans = ans >= 0
        ans = ans <= 1
        return ans
    model.x = Var(model.B, initialize=2)
    model.c = Constraint(rule=f)
    self.assertEqual(model.c(), 8)
    self.assertEqual(value(model.c.body), 8)