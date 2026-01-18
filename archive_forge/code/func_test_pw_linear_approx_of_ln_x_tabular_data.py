from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
def test_pw_linear_approx_of_ln_x_tabular_data(self):
    m = self.make_ln_x_model()
    m.pw = PiecewiseLinearFunction(tabular_data={1: 0, 3: log(3), 6: log(6), 10: log(10)})
    self.check_ln_x_approx(m.pw, m.x)