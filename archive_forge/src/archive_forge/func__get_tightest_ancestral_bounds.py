from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.modeling import unique_component_name
from pyomo.core import (
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core.expr import identify_variables
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.util import is_child_of, get_gdp_tree
from pyomo.repn.standard_repn import generate_standard_repn
import logging
def _get_tightest_ancestral_bounds(self, v_bounds, disjunct, gdp_forest):
    lb = None
    ub = None
    parent = disjunct
    while lb is None or ub is None:
        if parent in v_bounds:
            l, u = v_bounds[parent]
            if lb is None and l is not None:
                lb = l
            if ub is None and u is not None:
                ub = u
        if parent is None:
            break
        parent = gdp_forest.parent_disjunct(parent)
    return (lb, ub)