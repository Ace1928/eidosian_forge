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
def _handle_product_constant_constant(visitor, node, arg1, arg2):
    _, arg1 = arg1
    _, arg2 = arg2
    ans = arg1 * arg2
    if ans != ans:
        if not arg1 or not arg2:
            deprecation_warning(f'Encountered {str(arg1)}*{str(arg2)} in expression tree.  Mapping the NaN result to 0 for compatibility with the lp_v1 writer.  In the future, this NaN will be preserved/emitted to comply with IEEE-754.', version='6.6.0')
            return (_, 0)
    return (_, arg1 * arg2)