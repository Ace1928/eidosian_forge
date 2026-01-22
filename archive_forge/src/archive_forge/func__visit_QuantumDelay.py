import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumDelay(self, node: ast.QuantumDelay) -> None:
    self._start_line()
    self.stream.write('delay[')
    self.visit(node.duration)
    self.stream.write('] ')
    self._visit_sequence(node.qubits, separator=', ')
    self._end_statement()