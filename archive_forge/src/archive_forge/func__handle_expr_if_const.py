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
def _handle_expr_if_const(visitor, node, arg1, arg2, arg3):
    _type, _test = arg1
    assert _type is _CONSTANT
    if _test:
        if _test != _test or _test.__class__ is InvalidNumber:
            return _handle_expr_if_nonlinear(visitor, node, arg1, arg2, arg3)
        return arg2
    else:
        return arg3