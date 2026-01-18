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
def visit_IndexNode(self, node):
    """
        Replace index nodes used to specialize cdef functions with fused
        argument types with the Attribute- or NameNode referring to the
        function. We then need to copy over the specialization properties to
        the attribute or name node.

        Because the indexing might be a Python indexing operation on a fused
        function, or (usually) a Cython indexing operation, we need to
        re-analyse the types.
        """
    self.visit_Node(node)
    if node.is_fused_index and (not node.type.is_error):
        node = node.base
    return node