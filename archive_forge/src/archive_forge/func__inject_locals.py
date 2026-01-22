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
def _inject_locals(self, node, func_name):
    lenv = self.current_env()
    entry = lenv.lookup_here(func_name)
    if entry:
        return node
    pos = node.pos
    if func_name in ('locals', 'vars'):
        if func_name == 'locals' and len(node.args) > 0:
            error(self.pos, "Builtin 'locals()' called with wrong number of args, expected 0, got %d" % len(node.args))
            return node
        elif func_name == 'vars':
            if len(node.args) > 1:
                error(self.pos, "Builtin 'vars()' called with wrong number of args, expected 0-1, got %d" % len(node.args))
            if len(node.args) > 0:
                return node
        return ExprNodes.LocalsExprNode(pos, self.current_scope_node(), lenv)
    else:
        if len(node.args) > 1:
            error(self.pos, "Builtin 'dir()' called with wrong number of args, expected 0-1, got %d" % len(node.args))
        if len(node.args) > 0:
            return node
        if lenv.is_py_class_scope or lenv.is_module_scope:
            if lenv.is_py_class_scope:
                pyclass = self.current_scope_node()
                locals_dict = ExprNodes.CloneNode(pyclass.dict)
            else:
                locals_dict = ExprNodes.GlobalsExprNode(pos)
            return ExprNodes.SortedDictKeysNode(locals_dict)
        local_names = sorted((var.name for var in lenv.entries.values() if var.name))
        items = [ExprNodes.IdentifierStringNode(pos, value=var) for var in local_names]
        return ExprNodes.ListNode(pos, args=items)