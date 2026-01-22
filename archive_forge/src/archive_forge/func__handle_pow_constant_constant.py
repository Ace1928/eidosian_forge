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
def _handle_pow_constant_constant(visitor, node, *args):
    arg1, arg2 = args
    ans = apply_node_operation(node, (arg1[1], arg2[1]))
    if ans.__class__ in native_complex_types:
        ans = complex_number_error(ans, visitor, node)
    return (_CONSTANT, ans)