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
class TemplateTransform(VisitorTransform):
    """
    Makes a copy of a template tree while doing substitutions.

    A dictionary "substitutions" should be passed in when calling
    the transform; mapping names to replacement nodes. Then replacement
    happens like this:
     - If an ExprStatNode contains a single NameNode, whose name is
       a key in the substitutions dictionary, the ExprStatNode is
       replaced with a copy of the tree given in the dictionary.
       It is the responsibility of the caller that the replacement
       node is a valid statement.
     - If a single NameNode is otherwise encountered, it is replaced
       if its name is listed in the substitutions dictionary in the
       same way. It is the responsibility of the caller to make sure
       that the replacement nodes is a valid expression.

    Also a list "temps" should be passed. Any names listed will
    be transformed into anonymous, temporary names.

    Currently supported for tempnames is:
    NameNode
    (various function and class definition nodes etc. should be added to this)

    Each replacement node gets the position of the substituted node
    recursively applied to every member node.
    """
    temp_name_counter = 0

    def __call__(self, node, substitutions, temps, pos):
        self.substitutions = substitutions
        self.pos = pos
        tempmap = {}
        temphandles = []
        for temp in temps:
            TemplateTransform.temp_name_counter += 1
            handle = UtilNodes.TempHandle(PyrexTypes.py_object_type)
            tempmap[temp] = handle
            temphandles.append(handle)
        self.tempmap = tempmap
        result = super(TemplateTransform, self).__call__(node)
        if temps:
            result = UtilNodes.TempsBlockNode(self.get_pos(node), temps=temphandles, body=result)
        return result

    def get_pos(self, node):
        if self.pos:
            return self.pos
        else:
            return node.pos

    def visit_Node(self, node):
        if node is None:
            return None
        else:
            c = node.clone_node()
            if self.pos is not None:
                c.pos = self.pos
            self.visitchildren(c)
            return c

    def try_substitution(self, node, key):
        sub = self.substitutions.get(key)
        if sub is not None:
            pos = self.pos
            if pos is None:
                pos = node.pos
            return ApplyPositionAndCopy(pos)(sub)
        else:
            return self.visit_Node(node)

    def visit_NameNode(self, node):
        temphandle = self.tempmap.get(node.name)
        if temphandle:
            return temphandle.ref(self.get_pos(node))
        else:
            return self.try_substitution(node, node.name)

    def visit_ExprStatNode(self, node):
        if isinstance(node.expr, NameNode):
            return self.try_substitution(node, node.expr.name)
        else:
            return self.visit_Node(node)