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
def test_transformation_block_nested_disjunction_outer_disjunction_target(self):
    """We should get identical behavior to the previous test if we
        specify the outer disjunction as the target"""
    m = models.makeBetweenStepsPaperExample_Nested()
    TransformationFactory('gdp.partition_disjuncts').apply_to(m, targets=m.disjunction, variable_partitions=[[m.disj1.x[1], m.disj1.x[2]], [m.disj1.x[3], m.disj1.x[4]]], compute_bounds_method=compute_fbbt_bounds)
    b, disj1, disj2 = self.check_transformation_block_indexed_var_on_disjunct(m, m.disjunction)
    self.check_transformation_block_nested_disjunction(m, disj2, m.disj1.x)