from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
def visit_ResultRefNode(self, node):
    expr = node.expression
    if expr is None or expr not in self._replacements:
        self.visitchildren(node)
        expr = node.expression
    if expr is not None:
        node.expression = self._replacements.get(expr, expr)
    return node