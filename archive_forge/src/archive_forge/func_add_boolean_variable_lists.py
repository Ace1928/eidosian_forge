from pyomo.core import (
from pyomo.core.base import TransformationFactory, Suffix, ConstraintList, Integers
from pyomo.core.base.block import Block, TraversalStrategy
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.discrete_problem_initialize import valid_init_strategies
from pyomo.contrib.gdpopt.util import (
from pyomo.gdp.disjunct import Disjunct, Disjunction
from pyomo.util.vars_from_expressions import get_vars_from_components
def add_boolean_variable_lists(util_block):
    util_block.boolean_variable_list = []
    util_block.non_indicator_boolean_variable_list = []
    for disjunct in util_block.disjunct_list:
        util_block.boolean_variable_list.append(disjunct.indicator_var)
    ind_var_set = ComponentSet(util_block.boolean_variable_list)
    for v in get_vars_from_components(util_block.parent_block(), ctype=LogicalConstraint, descend_into=(Block, Disjunct), active=True, sort=SortComponents.deterministic):
        if v not in ind_var_set:
            util_block.boolean_variable_list.append(v)
            util_block.non_indicator_boolean_variable_list.append(v)