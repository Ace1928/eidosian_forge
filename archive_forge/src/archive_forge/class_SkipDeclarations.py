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
class SkipDeclarations(object):
    """
    Variable and function declarations can often have a deep tree structure,
    and yet most transformations don't need to descend to this depth.

    Declaration nodes are removed after AnalyseDeclarationsTransform, so there
    is no need to use this for transformations after that point.
    """

    def visit_CTypeDefNode(self, node):
        return node

    def visit_CVarDefNode(self, node):
        return node

    def visit_CDeclaratorNode(self, node):
        return node

    def visit_CBaseTypeNode(self, node):
        return node

    def visit_CEnumDefNode(self, node):
        return node

    def visit_CStructOrUnionDefNode(self, node):
        return node

    def visit_CppClassNode(self, node):
        if node.visibility != 'extern':
            self.visitchildren(node)
        return node