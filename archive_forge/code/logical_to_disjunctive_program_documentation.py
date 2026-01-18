from pyomo.contrib.cp.transform.logical_to_disjunctive_walker import (
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.core import (
from pyomo.core.base.block import _BlockData
from pyomo.core.base import SortComponents
from pyomo.core.util import target_list
from pyomo.gdp import Disjunct, Disjunction

    Re-encode logical constraints as linear constraints,
    converting Boolean variables to binary.
    