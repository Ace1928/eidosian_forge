import itertools
import logging
import sys
import builtins
from contextlib import nullcontext
from pyomo.common.errors import TemplateExpressionError
from pyomo.core.expr.base import ExpressionBase, ExpressionArgs_Mixin, NPV_Mixin
from pyomo.core.expr.logical_expr import BooleanExpression
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.numvalue import (
from pyomo.core.expr.relational_expr import tuple_to_relational_expr
from pyomo.core.expr.visitor import (
def _set_iter_vals(self, val):
    for i, iterGroup in enumerate(self._tse._iters):
        if len(iterGroup) == 1:
            iterGroup[0].set_value(val[i], self._lock)
        else:
            for j, v in enumerate(val[i]):
                iterGroup[j].set_value(v, self._lock)