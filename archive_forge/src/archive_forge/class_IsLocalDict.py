from __future__ import annotations
import ast
import asyncio
import inspect
import textwrap
from functools import lru_cache
from inspect import signature
from itertools import groupby
from typing import (
from langchain_core.pydantic_v1 import BaseConfig, BaseModel
from langchain_core.pydantic_v1 import create_model as _create_model_base
from langchain_core.runnables.schema import StreamEvent
class IsLocalDict(ast.NodeVisitor):
    """Check if a name is a local dict."""

    def __init__(self, name: str, keys: Set[str]) -> None:
        self.name = name
        self.keys = keys

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if isinstance(node.ctx, ast.Load) and isinstance(node.value, ast.Name) and (node.value.id == self.name) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            self.keys.add(node.slice.value)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and (node.func.value.id == self.name) and (node.func.attr == 'get') and (len(node.args) in (1, 2)) and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            self.keys.add(node.args[0].value)