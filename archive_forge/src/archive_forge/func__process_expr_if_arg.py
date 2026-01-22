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
def _process_expr_if_arg(arg, kwargs, name):
    alt = kwargs.pop(name, None)
    if alt is not None:
        if arg is not None:
            raise ValueError(f'Cannot specify both {name}_ and {name}')
        arg = alt
    _type = _categorize_arg_type(arg)
    while _type < ARG_TYPE.INVALID:
        if _type is ARG_TYPE.MUTABLE:
            arg = _recast_mutable(arg)
        elif _type is ARG_TYPE.ASNUMERIC:
            arg = arg.as_numeric()
        else:
            raise DeveloperError('_categorize_arg_type() returned unexpected ARG_TYPE')
        _type = _categorize_arg_type(arg)
    return (arg, _type)