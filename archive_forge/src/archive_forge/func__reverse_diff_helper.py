from pyomo.common.collections import ComponentMap, ComponentSet
import pyomo.core.expr as _expr
from pyomo.core.expr.visitor import ExpressionValueVisitor, nonpyomo_leaf_types
from pyomo.core.expr.numvalue import value, is_constant
from pyomo.core.expr import exp, log, sin, cos
import math
def _reverse_diff_helper(expr, numeric=True):
    val_dict = ComponentMap()
    der_dict = ComponentMap()
    expr_list = list()
    visitorA = _LeafToRootVisitor(val_dict, der_dict, expr_list, numeric=numeric)
    visitorA.dfs_postorder_stack(expr)
    der_dict[expr] = 1
    for e in reversed(expr_list):
        if e.__class__ in _diff_map:
            _diff_map[e.__class__](e, val_dict, der_dict)
        elif e.is_named_expression_type():
            _diff_GeneralExpression(e, val_dict, der_dict)
        else:
            raise DifferentiationException('Unsupported expression type for differentiation: {0}'.format(type(e)))
    return der_dict