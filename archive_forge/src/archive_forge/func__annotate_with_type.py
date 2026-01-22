from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_with_type(self, expression: E, target_type: exp.DataType.Type) -> E:
    self._set_type(expression, target_type)
    return self._annotate_args(expression)