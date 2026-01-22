import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_BitType(self, _node: ast.BitType) -> None:
    self.stream.write('bit')