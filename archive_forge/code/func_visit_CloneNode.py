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
def visit_CloneNode(self, node):
    self._print_node(node)
    self.indent()
    line = node.pos[1]
    if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
        print('%s- %s: %s' % (self._indent, 'arg', self.repr_of(node.arg)))
    self.indent()
    self.visitchildren(node.arg)
    self.unindent()
    self.unindent()
    return node