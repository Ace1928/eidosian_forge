from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def visit_ExecStatNode(self, node):
    lenv = self.current_env()
    self.visitchildren(node)
    if len(node.args) == 1:
        node.args.append(ExprNodes.GlobalsExprNode(node.pos))
        if not lenv.is_module_scope:
            node.args.append(ExprNodes.LocalsExprNode(node.pos, self.current_scope_node(), lenv))
    return node