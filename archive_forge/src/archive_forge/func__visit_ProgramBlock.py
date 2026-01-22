import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_ProgramBlock(self, node: ast.ProgramBlock) -> None:
    self.stream.write('{\n')
    self._current_indent += 1
    for statement in node.statements:
        self.visit(statement)
    self._current_indent -= 1
    self._start_line()
    self.stream.write('}')