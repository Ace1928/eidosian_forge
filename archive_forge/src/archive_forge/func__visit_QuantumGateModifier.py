import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumGateModifier(self, node: ast.QuantumGateModifier) -> None:
    self.stream.write(self._MODIFIER_LOOKUP[node.modifier])
    if node.argument:
        self.stream.write('(')
        self.visit(node.argument)
        self.stream.write(')')