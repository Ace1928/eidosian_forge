import sys
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr import (
from pyomo.core.base.constraint import _GeneralConstraintData
def test_set_expr_explicit_univariate(self):
    """Test expr= option (univariate expression)"""
    model = ConcreteModel()
    model.x = Var(initialize=2)
    ans = model.x >= 0
    ans = ans <= 1
    model.c = Constraint(expr=ans)
    self.assertEqual(model.c(), 2)
    self.assertEqual(model.c.body(), 2)
    self.assertEqual(value(model.c.body), 2)