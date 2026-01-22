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
class _AssignmentExpressionTargetNameFinder(TreeVisitor):

    def __init__(self):
        super(_AssignmentExpressionTargetNameFinder, self).__init__()
        self.target_names = {}

    def find_target_names(self, target):
        if target.is_name:
            return [target.name]
        elif target.is_sequence_constructor:
            names = []
            for arg in target.args:
                names.extend(self.find_target_names(arg))
            return names
        return []

    def visit_ForInStatNode(self, node):
        self.target_names[node] = tuple(self.find_target_names(node.target))
        self.visitchildren(node)

    def visit_ComprehensionNode(self, node):
        pass

    def visit_LambdaNode(self, node):
        pass

    def visit_Node(self, node):
        self.visitchildren(node)