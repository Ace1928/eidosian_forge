from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def test_pw_linear_approx_of_ln_x_linear_funcs(self):
    m = self.make_ln_x_model()
    m.pw = PiecewiseLinearFunction(simplices=[(1, 3), (3, 6), (6, 10)], linear_functions=[m.f1, m.f2, m.f3])
    self.check_ln_x_approx(m.pw, m.x)