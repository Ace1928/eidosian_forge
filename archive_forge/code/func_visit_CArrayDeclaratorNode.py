from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_CArrayDeclaratorNode(self, node):
    self.visit(node.base)
    self.put(u'[')
    if node.dimension is not None:
        self.visit(node.dimension)
    self.put(u']')