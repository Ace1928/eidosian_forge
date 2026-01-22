from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
class NegatedStepFunction(StepFunction):
    """
    The negated form of an elementary step function: That is, it represents
    subtracting the elementary function's (nonnegative) height rather than
    adding it.

    Args:
       arg (Step or Pulse): Child elementary step function of this node
    """
    __slots__ = '_args_'

    def __init__(self, args):
        self._args_ = args

    def nargs(self):
        return 1

    def _to_string(self, values, verbose, smap):
        return '- %s' % values[0]