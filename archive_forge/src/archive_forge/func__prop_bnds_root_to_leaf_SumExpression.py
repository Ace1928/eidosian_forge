from collections import defaultdict
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.contrib.fbbt.expression_bounds_walker import ExpressionBoundsVisitor
import pyomo.core.expr.numeric_expr as numeric_expr
from pyomo.core.expr.visitor import (
from pyomo.core.expr.numvalue import nonpyomo_leaf_types, value
from pyomo.core.expr.numvalue import is_fixed
import pyomo.contrib.fbbt.interval as interval
import math
from pyomo.core.base.block import Block
from pyomo.core.base.constraint import Constraint
from pyomo.core.base.var import Var
from pyomo.gdp import Disjunct
from pyomo.core.base.expression import _GeneralExpressionData, ScalarExpression
import logging
from pyomo.common.errors import InfeasibleConstraintException, PyomoException
from pyomo.common.config import (
from pyomo.common.numeric_types import native_types
from the constraint, we know that 1 <= x*y + z <= 1, so we may 
def _prop_bnds_root_to_leaf_SumExpression(node, bnds_dict, feasibility_tol):
    """
    This function is a bit complicated. A simpler implementation
    would loop through each argument in the sum and do the following:

    bounds_on_arg_i = bounds_on_entire_sum - bounds_on_sum_of_args_excluding_arg_i

    and the bounds_on_sum_of_args_excluding_arg_i could be computed
    for each argument. However, the computational expense would grow
    approximately quadratically with the length of the sum. Thus,
    we do the following. Consider the expression

    y = x1 + x2 + x3 + x4

    and suppose we have bounds on y. We first accumulate bounds to
    obtain a list like the following

    [(x1)_bounds, (x1+x2)_bounds, (x1+x2+x3)_bounds, (x1+x2+x3+x4)_bounds]

    Then we can propagate bounds back to x1, x2, x3, and x4 with the
    following

    (x4)_bounds = (x1+x2+x3+x4)_bounds - (x1+x2+x3)_bounds
    (x3)_bounds = (x1+x2+x3)_bounds - (x1+x2)_bounds
    (x2)_bounds = (x1+x2)_bounds - (x1)_bounds

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ProductExpression
    bnds_dict: ComponentMap
    feasibility_tol: float
        If the bounds computed on the body of a constraint violate the bounds of the constraint by more than
        feasibility_tol, then the constraint is considered infeasible and an exception is raised. This tolerance
        is also used when performing certain interval arithmetic operations to ensure that none of the feasible
        region is removed due to floating point arithmetic and to prevent math domain errors (a larger value
        is more conservative).
    """
    bnds = (0, 0)
    accumulated_bounds = [bnds]
    for arg in node.args:
        bnds = interval.add(*bnds, *bnds_dict[arg])
        accumulated_bounds.append(bnds)
    lb0, ub0 = bnds_dict[node]
    if lb0 > bnds[0]:
        bnds = (lb0, bnds[1])
    if ub0 < bnds[1]:
        bnds = (bnds[0], ub0)
    accumulated_bounds[-1] = bnds
    lb0, ub0 = accumulated_bounds[-1]
    for i, arg in enumerate(reversed(node.args)):
        lb1, ub1 = accumulated_bounds[-2 - i]
        lb2, ub2 = bnds_dict[arg]
        _lb1, _ub1 = interval.sub(lb0, ub0, lb2, ub2)
        _lb2, _ub2 = interval.sub(lb0, ub0, lb1, ub1)
        if _lb1 > lb1:
            lb1 = _lb1
        if _ub1 < ub1:
            ub1 = _ub1
        if _lb2 > lb2:
            lb2 = _lb2
        if _ub2 < ub2:
            ub2 = _ub2
        lb0, ub0 = (lb1, ub1)
        bnds_dict[arg] = (lb2, ub2)