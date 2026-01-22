import inspect
import logging
import sys
from copy import deepcopy
from collections import deque
from pyomo.common.deprecation import deprecated, deprecation_warning
from pyomo.common.errors import DeveloperError, TemplateExpressionError
from pyomo.common.numeric_types import (
import pyomo.core.expr.expr_common as common
from pyomo.core.expr.symbol_map import SymbolMap
def _expression_is_fixed(node):
    """Return bool indicating if this expression is fixed (non-variable)

    Args:
        node: The root node of an expression tree.

    Returns: bool

    """
    visitor = _IsFixedVisitor()
    return visitor.dfs_postorder_stack(node)