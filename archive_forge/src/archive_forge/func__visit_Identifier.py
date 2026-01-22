import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Identifier(self, node: ast.Identifier) -> None:
    self.stream.write(node.string)