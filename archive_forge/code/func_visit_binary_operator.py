from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def visit_binary_operator(self, node: mparser.BinaryOperatorNode) -> None:
    node.left.accept(self)
    node.operator.accept(self)
    node.right.accept(self)
    if node.whitespaces:
        node.whitespaces.accept(self)