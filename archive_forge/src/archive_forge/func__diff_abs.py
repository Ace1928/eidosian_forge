from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _diff_abs(node, val_dict, der_dict):
    """
    Reverse automatic differentiation on the abs function.
    This will raise an exception at 0.

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    assert len(node.args) == 1
    arg = node.args[0]
    der = der_dict[node]
    val = val_dict[arg]
    if is_constant(val) and val == 0:
        raise DifferentiationException('Cannot differentiate abs(x) at x=0')
    der_dict[arg] += der * val / abs(val)