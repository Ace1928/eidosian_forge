import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_QuantumArgument(self, node: ast.QuantumArgument) -> None:
    self.stream.write('qubit')
    if node.designator:
        self.visit(node.designator)
    self.stream.write(' ')
    self.visit(node.identifier)