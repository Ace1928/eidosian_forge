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
class GetLambdaSource(ast.NodeVisitor):
    """Get the source code of a lambda function."""

    def __init__(self) -> None:
        """Initialize the visitor."""
        self.source: Optional[str] = None
        self.count = 0

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        """Visit a lambda function."""
        self.count += 1
        if hasattr(ast, 'unparse'):
            self.source = ast.unparse(node)