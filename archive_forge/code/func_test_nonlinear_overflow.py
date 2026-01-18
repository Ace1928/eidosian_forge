import logging
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.errors import IterationLimitError
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.core.expr.calculus.diff_with_sympy import differentiate_available
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.core.expr.sympy_tools import sympy_available
@unittest.skipUnless(differentiate_available, 'this test requires sympy')
def test_nonlinear_overflow(self):
    m = ConcreteModel()
    m.x = Var(initialize=1)
    m.c = Constraint(expr=exp(100.0 * m.x ** 2) == 100)
    calculate_variable_from_constraint(m.x, m.c)
    self.assertAlmostEqual(value(m.x), 0.214597, 5)