from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
def visit_TempsBlockNode(self, node):
    """
        Temporaries are output like $1_1', where the first number is
        an index of the TempsBlockNode and the second number is an index
        of the temporary which that block allocates.
        """
    idx = 0
    for handle in node.temps:
        self.tempnames[handle] = '$%d_%d' % (self.tempblockindex, idx)
        idx += 1
    self.tempblockindex += 1
    self.visit(node.body)