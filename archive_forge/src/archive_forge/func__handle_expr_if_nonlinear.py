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
def _handle_expr_if_nonlinear(visitor, node, arg1, arg2, arg3):
    ans = visitor.Result()
    ans.nonlinear = Expr_ifExpression((to_expression(visitor, arg1), to_expression(visitor, arg2), to_expression(visitor, arg3)))
    return (_GENERAL, ans)