from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
class PxdWriter(DeclarationWriter, ExpressionWriter):
    """
    A Cython code writer for everything supported in pxd files.
    (currently unused)
    """

    def __call__(self, node):
        print(u'\n'.join(self.write(node).lines))
        return node

    def visit_CFuncDefNode(self, node):
        if node.overridable:
            self.startline(u'cpdef ')
        else:
            self.startline(u'cdef ')
        if node.modifiers:
            self.put(' '.join(node.modifiers))
            self.put(' ')
        if node.visibility != 'private':
            self.put(node.visibility)
            self.put(u' ')
        if node.api:
            self.put(u'api ')
        self.visit(node.declarator)

    def visit_StatNode(self, node):
        pass