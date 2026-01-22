from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _parse_ordered(self, parse_method: t.Optional[t.Callable]=None) -> t.Optional[exp.Ordered]:
    asc = self._match(TokenType.PLUS)
    desc = self._match(TokenType.DASH) or (asc and False)
    term = term = super()._parse_ordered(parse_method=parse_method)
    if term and desc:
        term.set('desc', True)
        term.set('nulls_first', False)
    return term