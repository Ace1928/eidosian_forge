import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import (
def test_scalar_invalid_expr(self):
    m = ConcreteModel()
    m.x = Var()
    with self.assertRaisesRegex(ValueError, "Cannot assign InequalityExpression to 'obj': ScalarObjective components only allow numeric expression types."):
        m.obj = Objective(expr=m.x <= 0)