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
def visit_ExceptClauseNode(self, node):
    if node.is_except_as:
        del_target = Nodes.DelStatNode(node.pos, args=[ExprNodes.NameNode(node.target.pos, name=node.target.name)], ignore_nonexisting=True)
        node.body = Nodes.StatListNode(node.pos, stats=[Nodes.TryFinallyStatNode(node.pos, body=node.body, finally_clause=Nodes.StatListNode(node.pos, stats=[del_target]))])
    self.visitchildren(node)
    return node