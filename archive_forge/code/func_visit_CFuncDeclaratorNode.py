from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_CFuncDeclaratorNode(self, node):
    self.visit(node.base)
    self.put(u'(')
    self.comma_separated_list(node.args)
    self.endline(u')')