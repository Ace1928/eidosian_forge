from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _build_object_construct(args: t.List) -> t.Union[exp.StarMap, exp.Struct]:
    expression = parser.build_var_map(args)
    if isinstance(expression, exp.StarMap):
        return expression
    return exp.Struct(expressions=[exp.PropertyEQ(this=k, expression=v) for k, v in zip(expression.keys, expression.values)])