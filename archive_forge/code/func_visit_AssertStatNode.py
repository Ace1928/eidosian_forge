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
def visit_AssertStatNode(self, node):
    """Extract the exception raising into a RaiseStatNode to simplify GIL handling.
        """
    if node.exception is None:
        node.exception = Nodes.RaiseStatNode(node.pos, exc_type=ExprNodes.NameNode(node.pos, name=EncodedString('AssertionError')), exc_value=node.value, exc_tb=None, cause=None, builtin_exc_name='AssertionError', wrap_tuple_value=True)
        node.value = None
    self.visitchildren(node)
    return node