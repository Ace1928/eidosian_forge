import logging
import sys
from operator import itemgetter
from itertools import filterfalse
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.numeric_types import (
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor, _EvaluationVisitor
from pyomo.core.expr import is_fixed, value
from pyomo.core.base.expression import Expression
import pyomo.core.kernel as kernel
from pyomo.repn.util import (
def _handle_product_nonlinear(visitor, node, arg1, arg2):
    ans = visitor.Result()
    if not visitor.expand_nonlinear_products:
        ans.nonlinear = to_expression(visitor, arg1) * to_expression(visitor, arg2)
        return (_GENERAL, ans)
    _, x1 = arg1
    _, x2 = arg2
    ans.multiplier = x1.multiplier * x2.multiplier
    x1.multiplier = x2.multiplier = 1
    ans.constant = x1.constant * x2.constant
    if x2.constant:
        c = x2.constant
        if c == 1:
            ans.linear = dict(x1.linear)
        else:
            ans.linear = {vid: c * coef for vid, coef in x1.linear.items()}
    if x1.constant:
        _merge_dict(ans.linear, x1.constant, x2.linear)
    ans.nonlinear = 0
    if x1.constant and x2.nonlinear is not None:
        ans.nonlinear += x1.constant * x2.nonlinear
    if x1.nonlinear is not None:
        ans.nonlinear += x1.nonlinear * to_expression(visitor, arg2)
    if x1.linear:
        x1.constant = 0
        x1.nonlinear = None
        x2.constant = 0
        ans.nonlinear += to_expression(visitor, arg1) * to_expression(visitor, arg2)
    return (_GENERAL, ans)