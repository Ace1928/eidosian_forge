from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def add_constraints_by_disjunct(util_block):
    constraints_by_disjunct = util_block.constraints_by_disjunct = {}
    for disj in util_block.disjunct_list:
        cons_list = constraints_by_disjunct[disj] = []
        for cons in disj.component_data_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
            cons_list.append(cons)