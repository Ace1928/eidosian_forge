from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_CEnumDefItemNode(self, node):
    self.startline(node.name)
    if node.cname:
        self.put(u' "%s"' % node.cname)
    if node.value:
        self.put(u' = ')
        self.visit(node.value)
    self.endline()