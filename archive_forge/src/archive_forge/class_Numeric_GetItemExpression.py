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
class Numeric_GetItemExpression(GetItemExpression, NumericExpression):
    __slots__ = ()

    def nargs(self):
        return len(self._args_)

    def _compute_polynomial_degree(self, result):
        if any((x != 0 for x in result[1:])):
            return None
        ans = 0
        for x in result[0].values():
            if x.__class__ in nonpyomo_leaf_types or not hasattr(x, 'polynomial_degree'):
                continue
            tmp = x.polynomial_degree()
            if tmp is None:
                return None
            elif tmp > ans:
                ans = tmp
        return ans