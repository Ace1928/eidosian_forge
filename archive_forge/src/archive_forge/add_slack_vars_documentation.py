from pyomo.core import (
from pyomo.common.modeling import unique_component_name
from pyomo.core.plugins.transform.hierarchy import NonIsomorphicTransformation
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base import ComponentUID
from pyomo.core.base.constraint import _ConstraintData
from pyomo.common.deprecation import deprecation_warning
import logging

    This plugin adds slack variables to every constraint or to the constraints
    specified in targets.
    