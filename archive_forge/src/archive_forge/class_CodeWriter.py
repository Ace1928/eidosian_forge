from __future__ import absolute_import, print_function
from .Compiler.Visitor import TreeVisitor
from .Compiler.ExprNodes import *
from .Compiler.Nodes import CSimpleBaseTypeNode
class CodeWriter(StatementWriter, ExpressionWriter):
    """
    A complete Cython code writer.
    """