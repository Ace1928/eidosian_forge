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
def _synthesize_assignment(self, node, env):
    genv = env
    while genv.is_py_class_scope or genv.is_c_class_scope:
        genv = genv.outer_scope
    if genv.is_closure_scope:
        rhs = node.py_cfunc_node = ExprNodes.InnerFunctionNode(node.pos, def_node=node, pymethdef_cname=node.entry.pymethdef_cname, code_object=ExprNodes.CodeObjectNode(node))
    else:
        binding = self.current_directives.get('binding')
        rhs = ExprNodes.PyCFunctionNode.from_defnode(node, binding)
        node.code_object = rhs.code_object
        if node.is_generator:
            node.gbody.code_object = node.code_object
    if env.is_py_class_scope:
        rhs.binding = True
    node.is_cyfunction = rhs.binding
    return self._create_assignment(node, rhs, env)