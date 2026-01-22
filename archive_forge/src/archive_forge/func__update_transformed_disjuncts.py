from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import GDP_Error
def _update_transformed_disjuncts(self, disj, disjunct_set, disjunct_containers):
    parent = disj.parent_component()
    if parent.is_indexed():
        if parent not in disjunct_containers:
            disjunct_set.update((d for d in parent.values() if d.active))
            disjunct_containers.add(parent)
        disjunct_set.remove(disj)