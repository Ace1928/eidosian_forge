from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def visit_WhileStatNode(self, node):
    condition_block = self.flow.nextblock()
    next_block = self.flow.newblock()
    self.flow.loops.append(LoopDescr(next_block, condition_block))
    if node.condition:
        self._visit(node.condition)
    self.flow.nextblock()
    self._visit(node.body)
    self.flow.loops.pop()
    if self.flow.block:
        self.flow.block.add_child(condition_block)
        self.flow.block.add_child(next_block)
    if node.else_clause:
        self.flow.nextblock(parent=condition_block)
        self._visit(node.else_clause)
        if self.flow.block:
            self.flow.block.add_child(next_block)
    else:
        condition_block.add_child(next_block)
    if next_block.parents:
        self.flow.block = next_block
    else:
        self.flow.block = None
    return node