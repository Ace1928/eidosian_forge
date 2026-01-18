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
def visit_PyClassDefNode(self, node):
    pxd_def = self.scope.lookup(node.name)
    if pxd_def:
        if pxd_def.is_cclass:
            return self.visit_CClassDefNode(node.as_cclass(), pxd_def)
        elif not pxd_def.scope or not pxd_def.scope.is_builtin_scope:
            error(node.pos, "'%s' redeclared" % node.name)
            if pxd_def.pos:
                error(pxd_def.pos, 'previous declaration here')
            return None
    return node