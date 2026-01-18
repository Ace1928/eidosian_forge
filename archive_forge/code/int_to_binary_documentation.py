from math import floor, log
import logging
from pyomo.common.collections import ComponentSet
from pyomo.common.config import ConfigBlock, ConfigValue, In
from pyomo.core import (
from pyomo.core.plugins.transform.hierarchy import IsomorphicTransformation
from pyomo.gdp import Disjunct
from pyomo.core.expr import identify_variables
from pyomo.common.modeling import unique_component_name
Apply the transformation to the given model.