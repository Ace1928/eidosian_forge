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
def test_unbounded_expression_error(self):
    m = models.makeBetweenStepsPaperExample()
    for i in m.x:
        m.x[i].setub(None)
    self.assertRaisesRegex(GDP_Error, "Expression x\\[1\\]\\*x\\[1\\] from constraint 'disjunction_disjuncts\\[0\\].constraint\\[1\\]' is unbounded! Please ensure all variables that appear in the constraint are bounded or specify compute_bounds_method=compute_optimal_bounds if the expression is bounded by the global constraints.", TransformationFactory('gdp.partition_disjuncts').apply_to, m, variable_partitions=[[m.x[1]], [m.x[2]], [m.x[3], m.x[4]]], compute_bounds_method=compute_fbbt_bounds)