from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_DictComprehensionAppendNode(self, node):
    self.visit(node.key_expr)
    self.put(u': ')
    self.visit(node.value_expr)