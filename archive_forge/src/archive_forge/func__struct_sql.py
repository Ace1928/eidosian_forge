from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _struct_sql(self: DuckDB.Generator, expression: exp.Struct) -> str:
    args: t.List[str] = []
    for i, expr in enumerate(expression.expressions):
        if isinstance(expr, exp.PropertyEQ):
            key = expr.name
            value = expr.expression
        else:
            key = f'_{i}'
            value = expr
        args.append(f'{self.sql(exp.Literal.string(key))}: {self.sql(value)}')
    return f'{{{', '.join(args)}}}'