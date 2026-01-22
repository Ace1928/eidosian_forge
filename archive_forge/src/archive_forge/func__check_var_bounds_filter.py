from pyomo.core import (
from pyomo.core.base import TransformationFactory, _VarData
from pyomo.core.plugins.transform.hierarchy import Transformation
from pyomo.common.config import ConfigBlock, ConfigValue, NonNegativeFloat
from pyomo.common.modeling import unique_component_name
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.opt import TerminationCondition
import logging
def _check_var_bounds_filter(constraint):
    """Check if the constraint is already implied by the variable bounds"""
    min_lhs = 0
    for v, coef in constraint['map'].items():
        if coef > 0:
            if v.lb is None:
                return True
            min_lhs += coef * v.lb
        elif coef < 0:
            if v.ub is None:
                return True
            min_lhs += coef * v.ub
    if value(min_lhs) >= constraint['lower']:
        return False
    return True