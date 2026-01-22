from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _diff_ExternalFunctionExpression(node, val_dict, der_dict):
    """

    Parameters
    ----------
    node: pyomo.core.expr.numeric_expr.ExternalFunctionExpression
    val_dict: ComponentMap
    der_dict: ComponentMap
    """
    der = der_dict[node]
    vals = tuple((val_dict[i] for i in node.args))
    derivs = node._fcn.evaluate_fgh(vals, fgh=1)[1]
    for ndx, arg in enumerate(node.args):
        der_dict[arg] += der * derivs[ndx]