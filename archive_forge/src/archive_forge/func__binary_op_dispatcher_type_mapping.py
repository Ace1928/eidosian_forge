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
def _binary_op_dispatcher_type_mapping(dispatcher, updates):

    def _any_asnumeric(a, b):
        b = b.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _asnumeric_any(a, b):
        a = a.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _asnumeric_asnumeric(a, b):
        a = a.as_numeric()
        b = b.as_numeric()
        return dispatcher[a.__class__, b.__class__](a, b)

    def _any_mutable(a, b):
        b = _recast_mutable(b)
        return dispatcher[a.__class__, b.__class__](a, b)

    def _mutable_any(a, b):
        a = _recast_mutable(a)
        return dispatcher[a.__class__, b.__class__](a, b)

    def _mutable_mutable(a, b):
        if a is b:
            a = b = _recast_mutable(a)
        else:
            a = _recast_mutable(a)
            b = _recast_mutable(b)
        return dispatcher[a.__class__, b.__class__](a, b)
    mapping = {}
    mapping.update({(i, ARG_TYPE.ASNUMERIC): _any_asnumeric for i in ARG_TYPE})
    mapping.update({(ARG_TYPE.ASNUMERIC, i): _asnumeric_any for i in ARG_TYPE})
    mapping[ARG_TYPE.ASNUMERIC, ARG_TYPE.ASNUMERIC] = _asnumeric_asnumeric
    mapping.update({(i, ARG_TYPE.MUTABLE): _any_mutable for i in ARG_TYPE})
    mapping.update({(ARG_TYPE.MUTABLE, i): _mutable_any for i in ARG_TYPE})
    mapping[ARG_TYPE.MUTABLE, ARG_TYPE.MUTABLE] = _mutable_mutable
    mapping.update({(i, ARG_TYPE.INVALID): _invalid for i in ARG_TYPE})
    mapping.update({(ARG_TYPE.INVALID, i): _invalid for i in ARG_TYPE})
    mapping.update(updates)
    return mapping