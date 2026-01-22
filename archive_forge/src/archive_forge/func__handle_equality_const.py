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
def _handle_equality_const(visitor, node, arg1, arg2):
    args, causes = InvalidNumber.parse_args(arg1[1], arg2[1])
    try:
        ans = args[0] == args[1]
    except:
        ans = False
        causes.append(str(sys.exc_info()[1]))
    if causes:
        ans = InvalidNumber(ans, causes)
    return (_CONSTANT, ans)