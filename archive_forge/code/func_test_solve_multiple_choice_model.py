from math import sqrt
from pyomo.common.dependencies import scipy_available
import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
@unittest.skipUnless(scipy_available, 'scipy is not available')
@unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
@unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
def test_solve_multiple_choice_model(self):
    m = models.make_log_x_model()
    TransformationFactory('contrib.piecewise.multiple_choice').apply_to(m)
    SolverFactory('gurobi').solve(m)
    ct.check_log_x_model_soln(self, m)