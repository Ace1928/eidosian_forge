from pyomo.contrib.cp.interval_var import (
from pyomo.core.base.component import Component
from pyomo.core.expr.base import ExpressionBase
from pyomo.core.expr.logical_expr import BooleanExpression
def _generate_difference_expression(_self, _other):
    if isinstance(_self, CumulativeFunction):
        if isinstance(_other, CumulativeFunction):
            return _subtract_cumuls(_self, _other)
        elif isinstance(_other, StepFunction):
            return _subtract_cumul_and_unit(_self, _other)
    elif isinstance(_self, StepFunction):
        if isinstance(_other, CumulativeFunction):
            return _subtract_unit_and_cumul(_self, _other)
        elif isinstance(_other, StepFunction):
            return _subtract_two_units(_self, _other)
    raise TypeError('Cannot subtract object of class %s from object of class %s' % (_other.__class__, _self.__class__))