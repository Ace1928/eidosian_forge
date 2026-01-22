import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Unary(self, node: ast.Unary):
    self.stream.write(node.op.value)
    if isinstance(node.operand, (ast.Unary, ast.Binary)) and _BINDING_POWER[node.operand.op].left < _BINDING_POWER[node.op].right:
        self.stream.write('(')
        self.visit(node.operand)
        self.stream.write(')')
    else:
        self.visit(node.operand)