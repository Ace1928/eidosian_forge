import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_DefaultCase(self, _node: ast.DefaultCase) -> None:
    self.stream.write('default')