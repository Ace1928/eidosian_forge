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
def visit_CClassDefNode(self, node, pxd_def=None):
    if pxd_def is None:
        pxd_def = self.scope.lookup(node.class_name)
    if pxd_def:
        if not pxd_def.defined_in_pxd:
            return node
        outer_scope = self.scope
        self.scope = pxd_def.type.scope
    self.visitchildren(node)
    if pxd_def:
        self.scope = outer_scope
    return node