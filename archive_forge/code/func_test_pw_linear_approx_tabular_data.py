from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
@unittest.skipUnless(scipy_available, 'scipy is not available')
def test_pw_linear_approx_tabular_data(self):
    m = self.make_model()
    m.pw = PiecewiseLinearFunction(tabular_data={(0, 1): g(0, 1), (0, 4): g(0, 4), (0, 7): g(0, 7), (3, 1): g(3, 1), (3, 4): g(3, 4), (3, 7): g(3, 7)})
    self.check_pw_linear_approximation(m)