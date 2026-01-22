import collections
from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap
from pyomo.core.expr.expr_common import ExpressionType
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.expr.numeric_expr import NumericExpression
from pyomo.core.expr.relational_expr import RelationalExpression
import pyomo.core.expr as EXPR
from pyomo.core.base import (
import pyomo.core.base.boolean_var as BV
from pyomo.core.base.expression import ScalarExpression, _GeneralExpressionData
from pyomo.core.base.param import ScalarParam, _ParamData
from pyomo.core.base.var import ScalarVar, _GeneralVarData
from pyomo.gdp.disjunct import AutoLinkedBooleanVar, Disjunct, Disjunction
def _dispatch_boolean_var(visitor, node):
    if node not in visitor.boolean_to_binary_map:
        binary = node.get_associated_binary()
        if binary is not None:
            visitor.boolean_to_binary_map[node] = binary
        else:
            z = visitor.z_vars.add()
            visitor.boolean_to_binary_map[node] = z
            node.associate_binary_var(z)
    if node.fixed:
        visitor.boolean_to_binary_map[node].fixed = True
        visitor.boolean_to_binary_map[node].set_value(int(node.value) if node.value is not None else None, skip_validation=True)
    return (False, visitor.boolean_to_binary_map[node])