from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_SetNode(self, node):
    if len(node.subexpr_nodes()) > 0:
        self.emit_sequence(node, u'{}')
    else:
        self.put(u'set()')