import copy
from pyomo.core.expr.numeric_expr import (
from pyomo.core.expr.relational_expr import (
from pyomo.core.base.expression import Expression
from . import linear
from .linear import _merge_dict, to_expression
class QuadraticRepnVisitor(linear.LinearRepnVisitor):
    Result = QuadraticRepn
    exit_node_handlers = _exit_node_handlers
    exit_node_dispatcher = linear.ExitNodeDispatcher(linear._initialize_exit_node_dispatcher(_exit_node_handlers))
    max_exponential_expansion = 2