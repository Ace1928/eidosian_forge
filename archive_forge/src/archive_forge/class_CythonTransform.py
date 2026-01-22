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
class CythonTransform(VisitorTransform):
    """
    Certain common conventions and utilities for Cython transforms.

     - Sets up the context of the pipeline in self.context
     - Tracks directives in effect in self.current_directives
    """

    def __init__(self, context):
        super(CythonTransform, self).__init__()
        self.context = context

    def __call__(self, node):
        from .ModuleNode import ModuleNode
        if isinstance(node, ModuleNode):
            self.current_directives = node.directives
        return super(CythonTransform, self).__call__(node)

    def visit_CompilerDirectivesNode(self, node):
        old = self.current_directives
        self.current_directives = node.directives
        self._process_children(node)
        self.current_directives = old
        return node

    def visit_Node(self, node):
        self._process_children(node)
        return node