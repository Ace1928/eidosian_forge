import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumGateDefinition(self, node: ast.QuantumGateDefinition) -> None:
    self._start_line()
    self.stream.write('gate ')
    self.visit(node.quantumGateSignature)
    self.stream.write(' ')
    self.visit(node.quantumBlock)
    self._end_line()