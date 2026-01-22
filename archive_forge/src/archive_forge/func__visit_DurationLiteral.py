import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_DurationLiteral(self, node: ast.DurationLiteral) -> None:
    self.stream.write(f'{node.value}{node.unit.value}')