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
def _mutable_mutable(a, b):
    if a is b:
        a = b = _recast_mutable(a)
    else:
        a = _recast_mutable(a)
        b = _recast_mutable(b)
    return dispatcher[a.__class__, b.__class__](a, b)