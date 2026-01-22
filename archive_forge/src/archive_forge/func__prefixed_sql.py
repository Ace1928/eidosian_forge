from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _prefixed_sql(self, prefix: str, expression: exp.Expression, arg: str) -> str:
    sql = self.sql(expression, arg)
    return f' {prefix} {sql}' if sql else ''