import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
def _register_new_iadd_mutablesum_handler(a, b):
    types = _categorize_arg_types(b)
    _iadd_mutablesum_dispatcher[b.__class__] = handler = _iadd_mutablesum_type_handler_mapping[types[0]]
    return handler(a, b)