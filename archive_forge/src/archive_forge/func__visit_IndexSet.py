import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_IndexSet(self, node: ast.IndexSet) -> None:
    self._visit_sequence(node.values, start='{', separator=', ', end='}')