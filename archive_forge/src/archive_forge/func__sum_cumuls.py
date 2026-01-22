from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
def _sum_cumuls(_self, _other):
    if _self.nargs() == len(_self._args_):
        _self._args_.extend(_other.args)
        return CumulativeFunction(_self._args_, nargs=len(_self._args_))
    else:
        return CumulativeFunction(_self.args + _other.args)