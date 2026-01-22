import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_BooleanLiteral(self, node: ast.BooleanLiteral):
    self.stream.write('true' if node.value else 'false')