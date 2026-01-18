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
def xbfs_yield_leaves(self, node):
    """
        Breadth-first search of an expression tree, except that
        leaf nodes are immediately visited.

        Note:
            This method has the same functionality as the
            PyUtilib :class:`SimpleVisitor.xbfs_yield_leaves <pyutilib.misc.visitor.SimpleVisitor.xbfs_yield_leaves>`
            method.  The difference is that this method
            is tailored to efficiently walk Pyomo expression trees.

        Args:
            node: The root node of the expression tree
                that is searched.

        Returns:
            The return value is determined by the :func:`finalize` function,
            which may be defined by the user.  Defaults to :const:`None`.
        """
    if node.__class__ in nonpyomo_leaf_types or not node.is_expression_type() or node.nargs() == 0:
        ans = self.visit(node)
        if not ans is None:
            yield ans
        return
    dq = deque([node])
    while dq:
        current = dq.popleft()
        for c in current.args:
            if c.__class__ in nonpyomo_leaf_types or not c.is_expression_type() or c.nargs() == 0:
                ans = self.visit(c)
                if not ans is None:
                    yield ans
            else:
                dq.append(c)