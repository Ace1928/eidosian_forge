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
def _transform_disjunction(self, disjunction, instance, bound_dict, transformation_blocks):
    disjunctions_to_transform = set()
    gdp_forest = get_gdp_tree((disjunction,), instance)
    for d in gdp_forest.topological_sort():
        if d.ctype is Disjunct:
            self._update_bounds_from_constraints(d, bound_dict, gdp_forest)
    self._create_transformation_constraints(disjunction, bound_dict, gdp_forest, transformation_blocks)