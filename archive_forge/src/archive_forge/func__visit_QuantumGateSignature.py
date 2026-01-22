import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumGateSignature(self, node: ast.QuantumGateSignature) -> None:
    self.visit(node.name)
    if node.params:
        self._visit_sequence(node.params, start='(', end=')', separator=', ')
    self.stream.write(' ')
    self._visit_sequence(node.qargList, separator=', ')