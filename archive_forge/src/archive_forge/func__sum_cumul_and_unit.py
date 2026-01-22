from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
def _sum_cumul_and_unit(_cumul, _unit):
    if _cumul.nargs() == len(_cumul._args_):
        _cumul._args_.append(_unit)
        return CumulativeFunction(_cumul._args_, nargs=len(_cumul._args_))
    else:
        return CumulativeFunction(_cumul.args + [_unit])