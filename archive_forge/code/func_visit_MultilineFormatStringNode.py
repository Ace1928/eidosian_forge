from __future__ import annotations
from .. import mparser
from .visitor import AstVisitor
from itertools import zip_longest
import re
import typing as T
def visit_MultilineFormatStringNode(self, node: mparser.MultilineFormatStringNode) -> None:
    self.result += 'f'
    self.visit_MultilineStringNode(node)