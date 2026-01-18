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
def visit_CallNode(self, node):
    self.visitchild(node, 'function')
    if not self.parallel_directive:
        self.visitchildren(node, exclude=('function',))
        return node
    if isinstance(node, ExprNodes.GeneralCallNode):
        args = node.positional_args.args
        kwargs = node.keyword_args
    else:
        args = node.args
        kwargs = {}
    parallel_directive_class = self.get_directive_class_node(node)
    if parallel_directive_class:
        node = parallel_directive_class(node.pos, args=args, kwargs=kwargs)
    return node