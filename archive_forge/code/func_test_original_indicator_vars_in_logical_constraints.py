import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.core.expr.logical_expr import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error, check_model_algebraic
from pyomo.gdp.plugins.partition_disjuncts import (
from pyomo.core import Block, value
import pyomo.core.expr as EXPR
import pyomo.gdp.tests.common_tests as ct
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.opt import check_available_solvers
@unittest.skipIf('gurobi_direct' not in solvers, 'Gurobi direct solver not available')
def test_original_indicator_vars_in_logical_constraints(self):
    m = models.makeLogicalConstraintsOnDisjuncts()
    TransformationFactory('gdp.between_steps').apply_to(m, variable_partitions=[[m.x]], compute_bounds_method=compute_fbbt_bounds)
    self.assertTrue(check_model_algebraic(m))
    SolverFactory('gurobi_direct').solve(m)
    self.assertAlmostEqual(value(m.x), 8)
    self.assertFalse(value(m.d[1].indicator_var))
    self.assertTrue(value(m.d[2].indicator_var))
    self.assertTrue(value(m.d[3].indicator_var))
    self.assertFalse(value(m.d[4].indicator_var))