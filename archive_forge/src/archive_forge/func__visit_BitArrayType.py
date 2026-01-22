import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_BitArrayType(self, node: ast.BitArrayType) -> None:
    self.stream.write(f'bit[{node.size}]')