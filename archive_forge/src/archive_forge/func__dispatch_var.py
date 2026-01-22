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
def _dispatch_var(visitor, node):
    return (False, node)