from __future__ import absolute_import
import re
from io import StringIO
from .Scanning import PyrexScanner, StringSourceDescriptor
from .Symtab import ModuleScope
from . import PyrexTypes
from .Visitor import VisitorTransform
from .Nodes import Node, StatListNode
from .ExprNodes import NameNode
from .StringEncoding import _unicode
from . import Parsing
from . import Main
from . import UtilNodes
def visit_ExprStatNode(self, node):
    if isinstance(node.expr, NameNode):
        return self.try_substitution(node, node.expr.name)
    else:
        return self.visit_Node(node)