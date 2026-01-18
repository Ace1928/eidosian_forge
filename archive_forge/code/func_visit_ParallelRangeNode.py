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
def visit_ParallelRangeNode(self, node):
    if node.nogil or self.nogil_declarator_only:
        node_was_nogil, node.nogil = (node.nogil, False)
        node = Nodes.GILStatNode(node.pos, state='nogil', body=node)
        if not node_was_nogil and self.nogil_declarator_only:
            node.scope_gil_state_known = False
        return self.visit_GILStatNode(node)
    if not self.nogil:
        error(node.pos, 'prange() can only be used without the GIL')
        return None
    node.nogil_check(self.env_stack[-1])
    self.visitchildren(node)
    return node