import collections
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from pyomo.core.expr import (
from typing import List
from pyomo.common.collections import Sequence
from pyomo.common.errors import PyomoException
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import native_types
class PrefixVisitor(StreamBasedExpressionVisitor):

    def __init__(self, include_named_exprs=True):
        super().__init__()
        self._result = None
        self._include_named_exprs = include_named_exprs

    def initializeWalker(self, expr):
        self._result = []
        return (True, None)

    def enterNode(self, node):
        ntype = type(node)
        if ntype in nonpyomo_leaf_types:
            self._result.append(node)
            return (tuple(), None)
        if node.is_expression_type():
            if node.is_named_expression_type():
                return (handle_named_expression(node, self._result, self._include_named_exprs), None)
            else:
                return (handler[ntype](node, self._result), None)
        else:
            self._result.append(node)
            return (tuple(), None)

    def finalizeResult(self, result):
        ans = self._result
        self._result = None
        return ans