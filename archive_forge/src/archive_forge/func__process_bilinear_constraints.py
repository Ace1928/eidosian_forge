import logging
import textwrap
from math import fabs
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.preprocessing.util import SuppressConstantObjectiveWarning
from pyomo.core import (
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import TerminationCondition as tc
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
def _process_bilinear_constraints(block, v1, v2, var_values, bilinear_constrs):
    if not (v2.has_lb() and v2.has_ub()):
        logger.warning(textwrap.dedent('            Attempting to transform bilinear term {v1} * {v2} using effectively\n            discrete variable {v1}, but {v2} is missing a lower or upper bound:\n            ({v2lb}, {v2ub}).\n            '.format(v1=v1, v2=v2, v2lb=v2.lb, v2ub=v2.ub)).strip())
        return False
    blk = Block()
    unique_name = unique_component_name(block, ('%s_%s_bilinear' % (v1.local_name, v2.local_name)).replace('[', '').replace(']', ''))
    block._induced_linearity_info.add_component(unique_name, blk)
    blk.valid_values = Set(initialize=sorted(var_values))
    blk.x_active = Var(blk.valid_values, domain=Binary, initialize=1)
    blk.v_increment = Var(blk.valid_values, domain=v2.domain, bounds=(v2.lb, v2.ub), initialize=v2.value)
    blk.v_defn = Constraint(expr=v2 == summation(blk.v_increment))

    @blk.Constraint(blk.valid_values)
    def v_lb(blk, val):
        return v2.lb * blk.x_active[val] <= blk.v_increment[val]

    @blk.Constraint(blk.valid_values)
    def v_ub(blk, val):
        return blk.v_increment[val] <= v2.ub * blk.x_active[val]
    blk.select_one_value = Constraint(expr=summation(blk.x_active) == 1)
    for bilinear_constr in bilinear_constrs:
        pass
        _reformulate_case_2(blk, v1, v2, bilinear_constr)
    pass