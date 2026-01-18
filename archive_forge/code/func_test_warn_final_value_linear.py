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
def test_warn_final_value_linear(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0, 1))
    m.c1 = Constraint(expr=m.x == 10)
    m.c2 = Constraint(expr=5 * m.x == 10)
    with LoggingIntercept() as LOG:
        calculate_variable_from_constraint(m.x, m.c1)
    self.assertEqual(LOG.getvalue().strip(), "Setting Var 'x' to a numeric value `10` outside the bounds (0, 1).")
    self.assertEqual(value(m.x), 10)
    with LoggingIntercept() as LOG:
        calculate_variable_from_constraint(m.x, m.c2)
    self.assertEqual(LOG.getvalue().strip(), "Setting Var 'x' to a numeric value `2.0` outside the bounds (0, 1).")
    self.assertEqual(value(m.x), 2)