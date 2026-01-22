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
def _handle_pow_ANY_constant(visitor, node, arg1, arg2):
    _, exp = arg2
    if exp == 1:
        return arg1
    elif exp > 1 and exp <= visitor.max_exponential_expansion and (int(exp) == exp):
        _type, _arg = arg1
        ans = (_type, _arg.duplicate())
        for i in range(1, int(exp)):
            ans = visitor.exit_node_dispatcher[ProductExpression, ans[0], _type](visitor, None, ans, (_type, _arg.duplicate()))
        return ans
    elif exp == 0:
        return (_CONSTANT, 1)
    else:
        return _handle_pow_nonlinear(visitor, node, arg1, arg2)