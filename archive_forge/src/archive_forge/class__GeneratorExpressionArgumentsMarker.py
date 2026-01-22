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
class _GeneratorExpressionArgumentsMarker(TreeVisitor, SkipDeclarations):

    def __init__(self, gen_expr):
        super(_GeneratorExpressionArgumentsMarker, self).__init__()
        self.gen_expr = gen_expr

    def visit_ExprNode(self, node):
        if not node.is_literal:
            assert not node.generator_arg_tag
            node.generator_arg_tag = self.gen_expr
        self.visitchildren(node)

    def visit_Node(self, node):
        return

    def visit_GeneratorExpressionNode(self, node):
        node.generator_arg_tag = self.gen_expr