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
def _unary_op_dispatcher_type_mapping(dispatcher, updates):

    def _asnumeric(a):
        a = a.as_numeric()
        return dispatcher[a.__class__](a)

    def _mutable(a):
        a = _recast_mutable(a)
        return dispatcher[a.__class__](a)
    mapping = {ARG_TYPE.ASNUMERIC: _asnumeric, ARG_TYPE.MUTABLE: _mutable, ARG_TYPE.INVALID: _invalid}
    mapping.update(updates)
    return mapping