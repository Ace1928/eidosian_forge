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
def _calculate_missing_M_values(self, active_disjuncts, arg_Ms, transBlock, transformed_constraints):
    scratch_blocks = {}
    all_vars = list(self._get_all_var_objects(active_disjuncts))
    for disjunct, other_disjunct in itertools.product(active_disjuncts, active_disjuncts):
        if disjunct is other_disjunct:
            continue
        if id(other_disjunct) in scratch_blocks:
            scratch = scratch_blocks[id(other_disjunct)]
        else:
            scratch = scratch_blocks[id(other_disjunct)] = Block()
            other_disjunct.add_component(unique_component_name(other_disjunct, 'scratch'), scratch)
            scratch.obj = Objective(expr=0)
            for v in all_vars:
                ref = Reference(v)
                scratch.add_component(unique_component_name(scratch, v.name), ref)
        for constraint in disjunct.component_data_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
            if constraint in transformed_constraints:
                continue
            if (constraint, other_disjunct) in arg_Ms:
                lower_M, upper_M = _convert_M_to_tuple(arg_Ms[constraint, other_disjunct], constraint, other_disjunct)
                self.used_args[constraint, other_disjunct] = (lower_M, upper_M)
            else:
                lower_M, upper_M = (None, None)
            unsuccessful_solve_msg = "Unsuccessful solve to calculate M value to relax constraint '%s' on Disjunct '%s' when Disjunct '%s' is selected." % (constraint.name, disjunct.name, other_disjunct.name)
            if constraint.lower is not None and lower_M is None:
                if lower_M is None:
                    scratch.obj.expr = constraint.body - constraint.lower
                    scratch.obj.sense = minimize
                    lower_M = self._solve_disjunct_for_M(other_disjunct, scratch, unsuccessful_solve_msg)
            if constraint.upper is not None and upper_M is None:
                if upper_M is None:
                    scratch.obj.expr = constraint.body - constraint.upper
                    scratch.obj.sense = maximize
                    upper_M = self._solve_disjunct_for_M(other_disjunct, scratch, unsuccessful_solve_msg)
            arg_Ms[constraint, other_disjunct] = (lower_M, upper_M)
            transBlock._mbm_values[constraint, other_disjunct] = (lower_M, upper_M)
    for blk in scratch_blocks.values():
        blk.parent_block().del_component(blk)
    return arg_Ms