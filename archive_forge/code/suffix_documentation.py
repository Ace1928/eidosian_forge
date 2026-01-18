import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.deprecation import deprecated
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readonly_property
from pyomo.core.kernel.dict_container import DictContainer
from pyomo.core.kernel.container_utils import define_homogeneous_container_type
Set the suffix direction.