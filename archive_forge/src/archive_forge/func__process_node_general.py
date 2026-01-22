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
def _process_node_general(self, node, recursion_limit):
    """Recursive routine for processing nodes with general callbacks

        This is the "general" implementation of the
        StreamBasedExpressionVisitor node processor that can handle any
        combination of registered callback functions.

        """
    if not recursion_limit:
        recursion_limit = self._compute_actual_recursion_limit()
    else:
        recursion_limit -= 1
    if self.enterNode is not None:
        tmp = self.enterNode(node)
        if tmp is None:
            args = data = None
        else:
            args, data = tmp
    else:
        args = None
        data = []
    if args is None:
        if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
            args = ()
        else:
            args = node.args
    context_manager = hasattr(args, '__enter__')
    if context_manager:
        args.__enter__()
    try:
        descend = True
        child_idx = -1
        arg_iter = iter(args)
        for child in arg_iter:
            child_idx += 1
            if self.beforeChild is not None:
                tmp = self.beforeChild(node, child, child_idx)
                if tmp is None:
                    descend = True
                else:
                    descend, child_result = tmp
            if descend:
                child_result = self._process_node(child, recursion_limit)
            if self.acceptChildResult is not None:
                data = self.acceptChildResult(node, data, child_result, child_idx)
            elif data is not None:
                data.append(child_result)
            if self.afterChild is not None:
                self.afterChild(node, child, child_idx)
    except RevertToNonrecursive:
        self._recursive_frame_to_nonrecursive_stack(locals())
        context_manager = False
        raise
    finally:
        if context_manager:
            args.__exit__(None, None, None)
    if self.exitNode is not None:
        return self.exitNode(node, data)
    else:
        return data