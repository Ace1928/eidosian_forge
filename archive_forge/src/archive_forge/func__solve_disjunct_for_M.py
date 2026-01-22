import itertools
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.gc_manager import PauseGC
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Reference, TransformationFactory
import pyomo.core.expr as EXPR
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.plugins.bigm_mixin import (
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import get_gdp_tree, _to_dict
from pyomo.network import Port
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.repn import generate_standard_repn
from weakref import ref as weakref_ref
def _solve_disjunct_for_M(self, other_disjunct, scratch_block, unsuccessful_solve_msg):
    solver = self._config.solver
    results = solver.solve(other_disjunct, load_solutions=False)
    if results.solver.termination_condition is TerminationCondition.infeasible:
        if any((s in solver.name for s in _trusted_solvers)):
            logger.debug("Disjunct '%s' is infeasible, deactivating." % other_disjunct.name)
            other_disjunct.deactivate()
            M = 0
        else:
            raise GDP_Error(unsuccessful_solve_msg)
    elif results.solver.termination_condition is not TerminationCondition.optimal:
        raise GDP_Error(unsuccessful_solve_msg)
    else:
        other_disjunct.solutions.load_from(results)
        M = value(scratch_block.obj.expr)
    return M