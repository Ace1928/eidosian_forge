from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _select_all(table: exp.Expression) -> t.Optional[exp.Select]:
    return exp.select('*').from_(table, copy=False) if table else None