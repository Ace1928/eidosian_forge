from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _diff_UnaryFunctionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.UnaryFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    if node.getname() in _unary_map:
        _unary_map[node.getname()](node, val_dict, der_dict)
    else:
        raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(node)))