import collections
import io
from typing import Sequence
from . import ast
from .experimental import ExperimentalFeatures
def _visit_Program(self, node: ast.Program) -> None:
    self.visit(node.header)
    for statement in node.statements:
        self.visit(statement)