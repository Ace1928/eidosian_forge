from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_bracket(self, expression: exp.Bracket) -> exp.Bracket:
    self._annotate_args(expression)
    bracket_arg = expression.expressions[0]
    this = expression.this
    if isinstance(bracket_arg, exp.Slice):
        self._set_type(expression, this.type)
    elif this.type.is_type(exp.DataType.Type.ARRAY):
        self._set_type(expression, seq_get(this.type.expressions, 0))
    elif isinstance(this, (exp.Map, exp.VarMap)) and bracket_arg in this.keys:
        index = this.keys.index(bracket_arg)
        value = seq_get(this.values, index)
        self._set_type(expression, value.type if value else None)
    else:
        self._set_type(expression, exp.DataType.Type.UNKNOWN)
    return expression