import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_set_expr_explicit_multivariate(self):
    """Test expr= option (multivariate expression)"""
    model = ConcreteModel()
    model.A = RangeSet(1, 4)
    model.x = Var(model.A, initialize=2)
    ans = 0
    for i in model.A:
        ans = ans + model.x[i]
    ans = ans >= 0
    ans = ans <= 1
    model.c = Constraint(expr=ans)
    self.assertEqual(model.c(), 8)
    self.assertEqual(model.c.body(), 8)
    self.assertEqual(value(model.c.body), 8)