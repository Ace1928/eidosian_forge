from pyomo.core import Constraint, Var, value
from math import fabs
import logging
from pyomo.common import deprecated
from pyomo.core.expr.visitor import identify_variables
from pyomo.util.blockutil import log_model_constraints
def _check_infeasible(obj, val, tol):
    if val is None:
        return 4
    infeasible = 0
    if obj.has_lb():
        lb = value(obj.lower, exception=False)
        if lb is None:
            infeasible |= 4 | 1
        elif lb - val > tol:
            infeasible |= 1
    if obj.has_ub():
        ub = value(obj.upper, exception=False)
        if ub is None:
            infeasible |= 4 | 2
        elif val - ub > tol:
            infeasible |= 2
    return infeasible