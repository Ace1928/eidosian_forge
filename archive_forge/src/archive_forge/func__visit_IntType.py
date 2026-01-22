import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_IntType(self, node: ast.IntType) -> None:
    self.stream.write('int')
    if node.size is not None:
        self.stream.write(f'[{node.size}]')