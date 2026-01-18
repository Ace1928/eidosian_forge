import pickle
import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types, as_numeric, value
from pyomo.core.expr.visitor import replace_expressions
from pyomo.repn import generate_standard_repn
from pyomo.environ import (
import pyomo.kernel
def test_product6(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    e = (m.x + m.y) * (m.x - m.y) * (m.x ** 2 + m.y ** 2)
    rep = generate_standard_repn(e)
    self.assertEqual(str(rep.to_expression()), '(x + y)*(x - y)*(x**2 + y**2)')
    self.assertTrue(rep.is_nonlinear())
    self.assertFalse(rep.is_quadratic())