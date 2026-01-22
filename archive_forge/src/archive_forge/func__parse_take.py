from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _parse_take(self, query: exp.Query) -> t.Optional[exp.Query]:
    num = self._parse_number()
    return query.limit(num) if num else None