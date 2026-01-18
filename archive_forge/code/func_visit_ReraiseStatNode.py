from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def visit_ReraiseStatNode(self, node):
    self.mark_position(node)
    if self.flow.exceptions:
        self.flow.block.add_child(self.flow.exceptions[-1].entry_point)
    self.flow.block = None
    return node