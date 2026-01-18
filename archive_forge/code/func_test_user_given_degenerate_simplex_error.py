from io import StringIO
import logging
import pickle
from pyomo.common.dependencies import attempt_import
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.core.expr.compare import (
from pyomo.environ import ConcreteModel, Constraint, log, Var
@unittest.skipUnless(numpy_available, 'numpy is not available')
def test_user_given_degenerate_simplex_error(self):
    m = self.make_model()
    with self.assertRaisesRegex(ValueError, 'When calculating the hyperplane approximation over the simplex with index 0, the matrix was unexpectedly singular. This likely means that this simplex is degenerate'):
        m.pw = PiecewiseLinearFunction(simplices=[((-2.0, 0.0, 1.0), (-2.0, 0.0, 4.0), (-2.0, 1.5, 1.0), (-2.0, 1.5, 4.0))], function=m.f)