import logging
from collections import defaultdict
from pyomo.common.autoslots import AutoSlots
import pyomo.common.config as cfg
from pyomo.common import deprecated
from pyomo.common.collections import ComponentMap, ComponentSet, DefaultComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.core.expr.numvalue import ZeroConstant
import pyomo.core.expr as EXPR
from pyomo.core.base import TransformationFactory, Reference
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.disjunct import _DisjunctData
from pyomo.gdp.plugins.gdp_to_mip_transformation import GDP_to_MIP_Transformation
from pyomo.gdp.transformed_disjunct import _TransformedDisjunct
from pyomo.gdp.util import (
from pyomo.core.util import target_list
from pyomo.util.vars_from_expressions import get_vars_from_components
from weakref import ref as weakref_ref
def _declare_disaggregated_var_bounds(self, original_var, disaggregatedVar, disjunct, bigmConstraint, lb_idx, ub_idx, var_free_indicator):
    lb = original_var.lb
    ub = original_var.ub
    if lb is None or ub is None:
        raise GDP_Error('Variables that appear in disjuncts must be bounded in order to use the hull transformation! Missing bound for %s.' % original_var.name)
    disaggregatedVar.setlb(min(0, lb))
    disaggregatedVar.setub(max(0, ub))
    if lb:
        bigmConstraint.add(lb_idx, var_free_indicator * lb <= disaggregatedVar)
    if ub:
        bigmConstraint.add(ub_idx, disaggregatedVar <= ub * var_free_indicator)
    original_var_info = original_var.parent_block().private_data()
    disaggregated_var_map = original_var_info.disaggregated_var_map
    disaggregated_var_info = disaggregatedVar.parent_block().private_data()
    disaggregated_var_map[disjunct][original_var] = disaggregatedVar
    disaggregated_var_info.original_var_map[disaggregatedVar] = original_var