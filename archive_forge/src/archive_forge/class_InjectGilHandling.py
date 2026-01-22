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
class InjectGilHandling(VisitorTransform, SkipDeclarations):
    """
    Allow certain Python operations inside of nogil blocks by implicitly acquiring the GIL.

    Must run before the AnalyseDeclarationsTransform to make sure the GILStatNodes get
    set up, parallel sections know that the GIL is acquired inside of them, etc.
    """
    nogil = False

    def _inject_gil_in_nogil(self, node):
        """Allow the (Python statement) node in nogil sections by wrapping it in a 'with gil' block."""
        if self.nogil:
            node = Nodes.GILStatNode(node.pos, state='gil', body=node)
        return node
    visit_RaiseStatNode = _inject_gil_in_nogil
    visit_PrintStatNode = _inject_gil_in_nogil

    def visit_GILStatNode(self, node):
        was_nogil = self.nogil
        self.nogil = node.state == 'nogil'
        self.visitchildren(node)
        self.nogil = was_nogil
        return node

    def visit_CFuncDefNode(self, node):
        was_nogil = self.nogil
        if isinstance(node.declarator, Nodes.CFuncDeclaratorNode):
            self.nogil = node.declarator.nogil and (not node.declarator.with_gil)
        self.visitchildren(node)
        self.nogil = was_nogil
        return node

    def visit_ParallelRangeNode(self, node):
        was_nogil = self.nogil
        self.nogil = node.nogil
        self.visitchildren(node)
        self.nogil = was_nogil
        return node

    def visit_ExprNode(self, node):
        return node
    visit_Node = VisitorTransform.recurse_to_children