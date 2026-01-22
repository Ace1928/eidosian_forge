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
def _mul_param_param(a, b):
    if a.is_constant():
        a = a.value
        if a in _zero_one_optimizations:
            return b if a else 0
        if b.is_constant():
            return a * b.value
    elif b.is_constant():
        b = b.value
        if b in _zero_one_optimizations:
            return a if b else 0
    return NPV_ProductExpression((a, b))