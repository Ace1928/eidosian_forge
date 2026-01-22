from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _maybe_annotate(self, expression: E) -> E:
    if id(expression) in self._visited:
        return expression
    annotator = self.annotators.get(expression.__class__)
    return annotator(self, expression) if annotator else self._annotate_with_type(expression, exp.DataType.Type.UNKNOWN)