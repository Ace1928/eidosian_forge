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
class AutoCpdefFunctionDefinitions(CythonTransform):

    def visit_ModuleNode(self, node):
        self.directives = node.directives
        self.imported_names = set()
        self.scope = node.scope
        self.visitchildren(node)
        return node

    def visit_DefNode(self, node):
        if self.scope.is_module_scope and self.directives['auto_cpdef'] and (node.name not in self.imported_names) and node.is_cdef_func_compatible():
            node = node.as_cfunction(scope=self.scope)
        return node

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

    def visit_FromImportStatNode(self, node):
        if self.scope.is_module_scope:
            for name, _ in node.items:
                self.imported_names.add(name)
        return node

    def visit_ExprNode(self, node):
        return node