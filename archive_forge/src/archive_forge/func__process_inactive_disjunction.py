import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
def _process_inactive_disjunction(self, djn):
    or_expr = '0'
    for disj in djn.disjuncts:
        iv = disj.binary_indicator_var
        label = self.add_var(iv)
        or_expr = '(+ ' + or_expr + ' ' + label + ')'
    if djn.xor:
        or_expr = '(assert (= 1 ' + or_expr + '))\n'
    else:
        or_expr = '(assert (>= 1 ' + or_expr + '))\n'
    self.expression_list.append(or_expr)