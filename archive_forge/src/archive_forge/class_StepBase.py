from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
class StepBase(StepFunction):
    __slots__ = '_args_'

    def __init__(self, args):
        self._args_ = [arg for arg in args]

    @property
    def _time(self):
        return self._args_[0]

    @property
    def _height(self):
        return self._args_[1]

    def nargs(self):
        return 2

    def _to_string(self, values, verbose, smap):
        return 'Step(%s, height=%s)' % (values[0], values[1])