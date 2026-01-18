import os
from os.path import abspath, dirname, join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.compare import assertExpressionsEqual
def test_sum2(self):
    model = ConcreteModel()
    model.A = Set(initialize=[1, 2, 3], doc='set A')
    model.x = Var(model.A)
    expr = quicksum((model.x[i] for i in model.x))
    baseline = 'x[1] + x[2] + x[3]'
    assertExpressionsEqual(self, expr, model.x[1] + model.x[2] + model.x[3])