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
def xbfs(self, node):
    """
        Breadth-first search of an expression tree,
        except that leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs <pyutilib.misc.visitor.SimpleVisitor.xbfs>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
    dq = deque([node])
    while dq:
        current = dq.popleft()
        self.visit(current)
        for c in current.args:
            if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                self.visit(c)
            else:
                dq.append(c)
    return self.finalize()