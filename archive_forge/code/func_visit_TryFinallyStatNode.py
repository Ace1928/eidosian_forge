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
def visit_TryFinallyStatNode(self, node):
    """
        Take care of try/finally statements in nogil code sections.
        """
    if not self.nogil or isinstance(node, Nodes.GILStatNode):
        return self.visit_Node(node)
    node.nogil_check = None
    node.is_try_finally_in_nogil = True
    self.visitchildren(node)
    return node