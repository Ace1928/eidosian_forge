import math
from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.gdp import Disjunction
def _add_bound(self, var):
    nm = self.variable_label_map.getSymbol(var)
    lb = var.lb
    ub = var.ub
    if lb is not None:
        self.bounds_list.append('(assert (>= ' + nm + ' ' + str(lb) + '))\n')
    if ub is not None:
        self.bounds_list.append('(assert (<= ' + nm + ' ' + str(ub) + '))\n')