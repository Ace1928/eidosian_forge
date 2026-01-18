from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_PrintStatNode(self, node):
    self.startline(u'print ')
    self.comma_separated_list(node.arg_tuple.args)
    if not node.append_newline:
        self.put(u',')
    self.endline()