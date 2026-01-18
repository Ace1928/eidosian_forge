from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def visit_AmpersandNode(self, node):
    if node.operand.is_name:
        self.mark_assignment(node.operand, fake_rhs_expr)
    self.visitchildren(node)
    return node