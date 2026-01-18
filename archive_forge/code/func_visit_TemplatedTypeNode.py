from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_TemplatedTypeNode(self, node):
    self.visit(node.base_type_node)
    self.put(u'[')
    self.comma_separated_list(node.positional_args + node.keyword_args.key_value_pairs)
    self.put(u']')