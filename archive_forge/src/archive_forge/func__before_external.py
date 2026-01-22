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
@staticmethod
def _before_external(visitor, child):
    ans = visitor.Result()
    if all((is_fixed(arg) for arg in child.args)):
        try:
            ans.constant = visitor.check_constant(visitor.evaluate(child), child)
            return (False, (_CONSTANT, ans))
        except:
            pass
    ans.nonlinear = child
    return (False, (_GENERAL, ans))