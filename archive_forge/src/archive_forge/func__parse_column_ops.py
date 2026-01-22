from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _parse_column_ops(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    this = super()._parse_column_ops(this)
    casts = []
    json_path = []
    while self._match(TokenType.COLON):
        path = super()._parse_column_ops(self._parse_field(any_token=True))
        while isinstance(path, exp.Cast):
            casts.append(path.to)
            path = path.this
        if path:
            json_path.append(path.sql(dialect='snowflake', copy=False))
    if json_path:
        this = self.expression(exp.JSONExtract, this=this, expression=self.dialect.to_json_path(exp.Literal.string('.'.join(json_path))))
        while casts:
            this = self.expression(exp.Cast, this=this, to=casts.pop())
    return this