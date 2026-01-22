import copy
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.base.expression import Expression
from . import linear
from .linear import _merge_dict, to_expression
def _handle_product_linear_linear(visitor, node, arg1, arg2):
    _, arg1 = arg1
    _, arg2 = arg2
    arg1.quadratic = _mul_linear_linear(visitor.var_order.__getitem__, arg1.linear, arg2.linear)
    if not arg2.constant:
        arg1.linear = {}
    elif arg2.constant != 1:
        c = arg2.constant
        _linear = arg1.linear
        for vid, coef in _linear.items():
            _linear[vid] = c * coef
    if arg1.constant:
        _merge_dict(arg1.linear, arg1.constant, arg2.linear)
    arg1.constant *= arg2.constant
    arg1.multiplier *= arg2.multiplier
    return (_QUADRATIC, arg1)