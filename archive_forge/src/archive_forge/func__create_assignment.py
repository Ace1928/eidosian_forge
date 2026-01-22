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
def _create_assignment(self, def_node, rhs, env):
    if def_node.decorators:
        for decorator in def_node.decorators[::-1]:
            rhs = ExprNodes.SimpleCallNode(decorator.pos, function=decorator.decorator, args=[rhs])
        def_node.decorators = None
    assmt = Nodes.SingleAssignmentNode(def_node.pos, lhs=ExprNodes.NameNode(def_node.pos, name=def_node.name), rhs=rhs)
    assmt.analyse_declarations(env)
    return assmt