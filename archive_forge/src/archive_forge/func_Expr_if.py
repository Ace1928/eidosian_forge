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
def Expr_if(IF_=None, THEN_=None, ELSE_=None, **kwargs):
    """
    Function used to construct a conditional numeric expression.

    This function accepts either of the following signatures:

       - Expr_if(IF={expr}, THEN={expr}, ELSE={expr})
       - Expr_if(IF_={expr}, THEN_={expr}, ELSE_={expr})

    (the former is historical, and the latter is required to support Cythonization)
    """
    _pv = False
    ELSE_, _type = _process_expr_if_arg(ELSE_, kwargs, 'ELSE')
    _pv |= _type >= ARG_TYPE.VAR or _type == ARG_TYPE.INVALID
    THEN_, _type = _process_expr_if_arg(THEN_, kwargs, 'THEN')
    _pv |= _type >= ARG_TYPE.VAR or _type == ARG_TYPE.INVALID
    IF_, _type = _process_expr_if_arg(IF_, kwargs, 'IF')
    _pv |= _type >= ARG_TYPE.VAR or _type == ARG_TYPE.INVALID
    if kwargs:
        raise ValueError('Unrecognized arguments: ' + ', '.join(kwargs))
    if _type is ARG_TYPE.NATIVE:
        return THEN_ if IF_ else ELSE_
    elif _type is ARG_TYPE.PARAM and IF_.is_constant():
        return THEN_ if IF_.value else ELSE_
    elif _pv:
        return Expr_ifExpression((IF_, THEN_, ELSE_))
    else:
        return NPV_Expr_ifExpression((IF_, THEN_, ELSE_))