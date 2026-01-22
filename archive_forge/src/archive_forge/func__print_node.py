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
def _print_node(self, node):
    line = node.pos[1]
    if self._line_range is None or self._line_range[0] <= line <= self._line_range[1]:
        if len(self.access_path) == 0:
            name = '(root)'
        else:
            parent, attr, idx = self.access_path[-1]
            if idx is not None:
                name = '%s[%d]' % (attr, idx)
            else:
                name = attr
        print('%s- %s: %s' % (self._indent, name, self.repr_of(node)))