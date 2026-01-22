from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
class CumulativeFunction(StepFunction):
    """
    A sum of elementary step functions (Pulse and Step), defining a step
    function over time. (Often used to model resource constraints.)

    Args:
        args (list or tuple): Child elementary step functions of this node
    """
    __slots__ = ('_args_', '_nargs')

    def __init__(self, args, nargs=None):
        self._args_ = [arg for arg in args]
        if nargs is None:
            self._nargs = len(args)
        else:
            self._nargs = nargs

    def nargs(self):
        return self._nargs

    def _to_string(self, values, verbose, smap):
        s = ''
        for i, arg in enumerate(self.args):
            if isinstance(arg, NegatedStepFunction):
                s += str(arg) + ' '
            else:
                s += '+ %s '[2 * (i == 0):] % str(arg)
        return s[:-1]