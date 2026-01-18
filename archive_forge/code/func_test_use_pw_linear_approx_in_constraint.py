from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def test_use_pw_linear_approx_in_constraint(self):
    m = self.make_model()

    def g1(x1, x2):
        return 3 * x1 + 5 * x2 - 4

    def g2(x1, x2):
        return 3 * x1 + 11 * x2 - 28
    m.pw = PiecewiseLinearFunction(simplices=self.simplices, linear_functions=[g1, g1, g2, g2])
    m.c = Constraint(expr=m.pw(m.x1, m.x2) <= 5)
    self.assertEqual(str(m.c.body.expr), 'pw(x1, x2)')
    self.assertIs(m.c.body.expr.pw_linear_function, m.pw)