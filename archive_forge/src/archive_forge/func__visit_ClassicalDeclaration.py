import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_ClassicalDeclaration(self, node: ast.ClassicalDeclaration) -> None:
    self._start_line()
    self.visit(node.type)
    self.stream.write(' ')
    self.visit(node.identifier)
    if node.initializer is not None:
        self.stream.write(' = ')
        self.visit(node.initializer)
    self._end_statement()