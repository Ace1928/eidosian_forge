import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_FloatType(self, node: ast.FloatType) -> None:
    self.stream.write(f'float[{self._FLOAT_WIDTH_LOOKUP[node]}]')