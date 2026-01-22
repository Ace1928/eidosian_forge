import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_BreakStatement(self, _node: ast.BreakStatement) -> None:
    self._write_statement('break')