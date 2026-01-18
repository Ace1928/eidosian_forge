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
def visit_ParallelWithBlockNode(self, node):
    if not self.nogil:
        error(node.pos, 'The parallel section may only be used without the GIL')
        return None
    if self.nogil_declarator_only:
        node = Nodes.GILStatNode(node.pos, state='nogil', body=node)
        node.scope_gil_state_known = False
        return self.visit_GILStatNode(node)
    if node.nogil_check:
        node.nogil_check(self.env_stack[-1])
    self.visitchildren(node)
    return node