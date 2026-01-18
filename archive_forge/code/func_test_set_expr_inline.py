import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_set_expr_inline(self):
    """Test expr= option (inline expression)"""
    model = ConcreteModel()
    model.A = RangeSet(1, 4)
    model.x = Var(model.A, initialize=2)
    model.c = Constraint(expr=(0, sum((model.x[i] for i in model.A)), 1))
    self.assertEqual(model.c(), 8)
    self.assertEqual(value(model.c.body), 8)