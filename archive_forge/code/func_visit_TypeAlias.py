from __future__ import annotations
import ast
import io
import keyword
import re
import sys
import token
import tokenize
from typing import Iterable
from coverage import env
from coverage.types import TLineNo, TSourceTokenLines
def visit_TypeAlias(self, node: ast.TypeAlias) -> None:
    """Invoked by ast.NodeVisitor.visit"""
    self.soft_key_lines.add(node.lineno)
    self.generic_visit(node)