from pyomo.gdp import GDP_Error
from pyomo.common.collections import ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.contrib.fbbt.interval as interval
from pyomo.core import Suffix
def _estimate_M(self, expr, constraint):
    expr_lb, expr_ub = self._expr_bound_visitor.walk_expression(expr)
    if expr_lb == -interval.inf or expr_ub == interval.inf:
        raise GDP_Error("Cannot estimate M for unbounded expressions.\n\t(found while processing constraint '%s'). Please specify a value of M or ensure all variables that appear in the constraint are bounded." % constraint.name)
    else:
        M = (expr_lb, expr_ub)
    return tuple(M)