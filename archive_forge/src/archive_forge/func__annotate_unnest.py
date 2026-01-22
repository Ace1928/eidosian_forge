from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_unnest(self, expression: exp.Unnest) -> exp.Unnest:
    self._annotate_args(expression)
    child = seq_get(expression.expressions, 0)
    if child and child.is_type(exp.DataType.Type.ARRAY):
        expr_type = seq_get(child.type.expressions, 0)
    else:
        expr_type = None
    self._set_type(expression, expr_type)
    return expression