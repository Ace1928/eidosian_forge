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
def test_warn_final_value_nonlinear(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.c3 = Constraint(expr=(m.x - 3.5) ** 2 == 0)
    with LoggingIntercept() as LOG:
        calculate_variable_from_constraint(m.x, m.c3)
    self.assertRegex(LOG.getvalue().strip(), "Setting Var 'x' to a numeric value `[0-9\\.]+` outside the bounds \\(0, 1\\).")
    self.assertAlmostEqual(value(m.x), 3.5, 3)
    m.x.domain = Binary
    with LoggingIntercept() as LOG:
        calculate_variable_from_constraint(m.x, m.c3)
    self.assertRegex(LOG.getvalue().strip(), "Setting Var 'x' to a value `[0-9\\.]+` \\(float\\) not in domain Binary.")
    self.assertAlmostEqual(value(m.x), 3.5, 3)