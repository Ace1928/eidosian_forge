from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
def visit_ContinueStatNode(self, node):
    if not self.flow.loops:
        return node
    loop = self.flow.loops[-1]
    self.mark_position(node)
    for exception in loop.exceptions[::-1]:
        if exception.finally_enter:
            self.flow.block.add_child(exception.finally_enter)
            if exception.finally_exit:
                exception.finally_exit.add_child(loop.loop_block)
            break
    else:
        self.flow.block.add_child(loop.loop_block)
    self.flow.block = None
    return node