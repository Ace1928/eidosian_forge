from __future__ import absolute_import
from .Errors import error, message
from . import ExprNodes
from . import Nodes
from . import Builtin
from . import PyrexTypes
from .. import Utils
from .PyrexTypes import py_object_type, unspecified_type
from .Visitor import CythonTransform, EnvTransform
def visit_ParallelStatNode(self, node):
    if self.parallel_block_stack:
        node.parent = self.parallel_block_stack[-1]
    else:
        node.parent = None
    nested = False
    if node.is_prange:
        if not node.parent:
            node.is_parallel = True
        else:
            node.is_parallel = node.parent.is_prange or not node.parent.is_parallel
            nested = node.parent.is_prange
    else:
        node.is_parallel = True
        nested = node.parent and node.parent.is_prange
    self.parallel_block_stack.append(node)
    nested = nested or len(self.parallel_block_stack) > 2
    if not self.parallel_errors and nested and (not node.is_prange):
        error(node.pos, 'Only prange() may be nested')
        self.parallel_errors = True
    if node.is_prange:
        child_attrs = node.child_attrs
        node.child_attrs = ['body', 'target', 'args']
        self.visitchildren(node)
        node.child_attrs = child_attrs
        self.parallel_block_stack.pop()
        if node.else_clause:
            node.else_clause = self.visit(node.else_clause)
    else:
        self.visitchildren(node)
        self.parallel_block_stack.pop()
    self.parallel_errors = False
    return node