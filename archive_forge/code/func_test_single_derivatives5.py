import pyomo.common.unittest as unittest
from pyomo.common.errors import DeveloperError, NondifferentiableError
from pyomo.environ import (
from pyomo.core.expr.calculus.diff_with_sympy import differentiate
from pyomo.core.expr.sympy_tools import (
def test_single_derivatives5(self):
    m = ConcreteModel()
    m.x = Var()
    m.y = Var()
    e = differentiate(m.x * m.y, wrt=m.x)
    self.assertIs(e, m.y)
    self.assertEqual(s(e), s(m.y))