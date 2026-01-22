from __future__ import absolute_import, print_function
import sys
import inspect
from . import TypeSlots
from . import Builtin
from . import Nodes
from . import ExprNodes
from . import Errors
from . import DebugFlags
from . import Future
import cython
class NodeFinder(TreeVisitor):
    """
    Find out if a node appears in a subtree.
    """

    def __init__(self, node):
        super(NodeFinder, self).__init__()
        self.node = node
        self.found = False

    def visit_Node(self, node):
        if self.found:
            pass
        elif node is self.node:
            self.found = True
        else:
            self._visitchildren(node, None, None)