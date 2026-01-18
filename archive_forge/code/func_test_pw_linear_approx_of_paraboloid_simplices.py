from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
@unittest.skipUnless(numpy_available, 'numpy are not available')
def test_pw_linear_approx_of_paraboloid_simplices(self):
    m = self.make_model()
    m.pw = PiecewiseLinearFunction(function=m.g, simplices=self.simplices)
    self.check_pw_linear_approximation(m)