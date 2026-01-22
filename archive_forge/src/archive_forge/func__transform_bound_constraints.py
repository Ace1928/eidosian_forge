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
def _transform_bound_constraints(self, active_disjuncts, transBlock, Ms):
    bounds_cons = ComponentMap()
    lower_bound_constraints_by_var = ComponentMap()
    upper_bound_constraints_by_var = ComponentMap()
    transformed_constraints = set()
    for disj in active_disjuncts:
        for c in disj.component_data_objects(Constraint, active=True, descend_into=Block, sort=SortComponents.deterministic):
            repn = generate_standard_repn(c.body)
            if repn.is_linear() and len(repn.linear_vars) == 1:
                v = repn.linear_vars[0]
                if v not in bounds_cons:
                    bounds_cons[v] = [{}, {}]
                M = [None, None]
                if c.lower is not None:
                    M[0] = (c.lower - repn.constant) / repn.linear_coefs[0]
                    if disj in bounds_cons[v][0]:
                        M[0] = max(M[0], bounds_cons[v][0][disj])
                    bounds_cons[v][0][disj] = M[0]
                    if v in lower_bound_constraints_by_var:
                        lower_bound_constraints_by_var[v].add((c, disj))
                    else:
                        lower_bound_constraints_by_var[v] = {(c, disj)}
                if c.upper is not None:
                    M[1] = (c.upper - repn.constant) / repn.linear_coefs[0]
                    if disj in bounds_cons[v][1]:
                        M[1] = min(M[1], bounds_cons[v][1][disj])
                    bounds_cons[v][1][disj] = M[1]
                    if v in upper_bound_constraints_by_var:
                        upper_bound_constraints_by_var[v].add((c, disj))
                    else:
                        upper_bound_constraints_by_var[v] = {(c, disj)}
                transBlock._mbm_values[c, disj] = M
                transformed_constraints.add(c)
    transformed = transBlock.transformed_bound_constraints
    offset = len(transformed)
    for i, (v, (lower_dict, upper_dict)) in enumerate(bounds_cons.items()):
        lower_rhs = 0
        upper_rhs = 0
        for disj in active_disjuncts:
            relaxationBlock = self._get_disjunct_transformation_block(disj, transBlock)
            if len(lower_dict) > 0:
                M = lower_dict.get(disj, None)
                if M is None:
                    M = v.lb
                if M is None:
                    raise GDP_Error("There is no lower bound for variable '%s', and Disjunct '%s' does not specify one in its constraints. The transformation cannot construct the special bound constraint relaxation without one of these." % (v.name, disj.name))
                lower_rhs += M * disj.indicator_var.get_associated_binary()
            if len(upper_dict) > 0:
                M = upper_dict.get(disj, None)
                if M is None:
                    M = v.ub
                if M is None:
                    raise GDP_Error("There is no upper bound for variable '%s', and Disjunct '%s' does not specify one in its constraints. The transformation cannot construct the special bound constraint relaxation without one of these." % (v.name, disj.name))
                upper_rhs += M * disj.indicator_var.get_associated_binary()
        idx = i + offset
        if len(lower_dict) > 0:
            transformed.add((idx, 'lb'), v >= lower_rhs)
            relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'lb']] = []
            for c, disj in lower_bound_constraints_by_var[v]:
                relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'lb']].append(c)
                disj.transformation_block._constraintMap['transformedConstraints'][c] = [transformed[idx, 'lb']]
        if len(upper_dict) > 0:
            transformed.add((idx, 'ub'), v <= upper_rhs)
            relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'ub']] = []
            for c, disj in upper_bound_constraints_by_var[v]:
                relaxationBlock._constraintMap['srcConstraints'][transformed[idx, 'ub']].append(c)
                if c in disj.transformation_block._constraintMap['transformedConstraints']:
                    disj.transformation_block._constraintMap['transformedConstraints'][c].append(transformed[idx, 'ub'])
                else:
                    disj.transformation_block._constraintMap['transformedConstraints'][c] = [transformed[idx, 'ub']]
    return transformed_constraints