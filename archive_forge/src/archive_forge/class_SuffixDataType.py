import enum
import logging
from pyomo.common.collections import ComponentMap
from pyomo.common.config import In
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.pyomo_typing import overload
from pyomo.common.timing import ConstructionTimer
from pyomo.core.base.component import ActiveComponent, ModelComponentFactory
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import Initializer
class SuffixDataType(enum.IntEnum):
    """Suffix data types

    AMPL only supports two data types for Suffixes: int and float.  The
    numeric values here are specific to the NL file format and should
    not be changed without checking/updating the NL writer.

    """
    INT = 0
    FLOAT = 4