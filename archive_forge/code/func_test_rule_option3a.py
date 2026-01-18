import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_rule_option3a(self):
    model = self.create_model()
    model.B = RangeSet(1, 4)

    @simple_constraint_rule
    def f(model, i):
        if i % 2 == 0:
            return None
        ans = 0
        for j in model.B:
            ans = ans + model.x[j]
        ans *= i
        ans = ans <= 0
        ans = ans >= 0
        return ans
    model.x = Var(model.B, initialize=2)
    model.c = Constraint(model.A, rule=f)
    self.assertEqual(model.c[1](), 8)
    self.assertEqual(len(model.c), 2)