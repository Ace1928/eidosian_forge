import logging
from pyomo.core.base import Constraint, Param, value, Suffix, Block
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.diffvar import DAE_Error
import pyomo.core.expr as EXPR
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.template_expr import IndexTemplate, _GetItemIndexer
from pyomo.common.dependencies import (
def _check_viewsumexpression(expr, i):
    """
    Accepts an equality expression and an index value. Checks the
    Sum expression at expr.arg(i) to see if it contains a
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>`. If so, return the
    GetItemExpression for the
    :py:class:`DerivativeVar<pyomo.dae.DerivativeVar>` and the RHS. If not,
    return None.
    """
    sumexp = expr.arg(i)
    items = []
    dv = None
    dvcoef = 1
    for idx, item in enumerate(sumexp.args):
        if dv is not None:
            items.append(item)
        elif isinstance(item, EXPR.GetItemExpression) and type(item.arg(0)) is DerivativeVar:
            dv = item
        elif type(item) is EXPR.ProductExpression:
            lhs = item.arg(0)
            rhs = item.arg(1)
            if (type(lhs) in native_numeric_types or not lhs.is_potentially_variable()) and (isinstance(rhs, EXPR.GetItemExpression) and type(rhs.arg(0)) is DerivativeVar):
                dv = rhs
                dvcoef = lhs
        else:
            items.append(item)
    if dv is not None:
        RHS = expr.arg(1 - i)
        for item in items:
            RHS -= item
        RHS = RHS / dvcoef
        return [dv, RHS]
    return None