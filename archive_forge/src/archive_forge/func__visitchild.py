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
@cython.final
def _visitchild(self, child, parent, attrname, idx):
    self.access_path.append((parent, attrname, idx))
    result = self._visit(child)
    self.access_path.pop()
    return result