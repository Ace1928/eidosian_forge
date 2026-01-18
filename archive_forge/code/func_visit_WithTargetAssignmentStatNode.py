from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def visit_WithTargetAssignmentStatNode(self, node):
    self.mark_assignment(node.lhs, node.with_node.enter_call)
    return node