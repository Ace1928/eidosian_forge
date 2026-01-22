from __future__ import annotations
import ast
import re
import typing as t
from dataclasses import dataclass
from string import Template
from types import CodeType
from urllib.parse import quote
from ..datastructures import iter_multi_items
from ..urls import _urlencode
from .converters import ValidationError
def _prefix_names(src: str) -> ast.stmt:
    """ast parse and prefix names with `.` to avoid collision with user vars"""
    tree = ast.parse(src).body[0]
    if isinstance(tree, ast.Expr):
        tree = tree.value
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = f'.{node.id}'
    return tree