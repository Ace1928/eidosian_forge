from pyomo.common.config import (
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.expr import differentiate
from pyomo.common.collections import ComponentSet
from pyomo.opt import SolverFactory
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import (
from pyomo.contrib.fme.fourier_motzkin_elimination import (
import logging
def _get_constraint_exprs(constraints, hull_to_bigm_map):
    """Returns a list of expressions which are constrain.expr translated
    into the bigm space, for each constraint in constraints.
    """
    cuts = []
    for cons in constraints:
        cuts.append(clone_without_expression_components(cons.expr, substitute=hull_to_bigm_map))
    return cuts