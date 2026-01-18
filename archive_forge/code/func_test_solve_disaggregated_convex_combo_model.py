import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var
@unittest.skipUnless(SolverFactory('gurobi').available(), 'Gurobi is not available')
@unittest.skipUnless(SolverFactory('gurobi').license_is_valid(), 'No license')
def test_solve_disaggregated_convex_combo_model(self):
    m = models.make_log_x_model()
    TransformationFactory('contrib.piecewise.disaggregated_convex_combination').apply_to(m)
    SolverFactory('gurobi').solve(m)
    ct.check_log_x_model_soln(self, m)