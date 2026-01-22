from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _annotate_to_map(self, expression: exp.ToMap) -> exp.ToMap:
    self._annotate_args(expression)
    map_type = exp.DataType(this=exp.DataType.Type.MAP)
    arg = expression.this
    if arg.is_type(exp.DataType.Type.STRUCT):
        for coldef in arg.type.expressions:
            kind = coldef.kind
            if kind != exp.DataType.Type.UNKNOWN:
                map_type.set('expressions', [exp.DataType.build('varchar'), kind])
                map_type.set('nested', True)
                break
    self._set_type(expression, map_type)
    return expression