from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _parse_order_by(self, query: exp.Select) -> t.Optional[exp.Query]:
    l_brace = self._match(TokenType.L_BRACE)
    expressions = self._parse_csv(self._parse_ordered)
    if l_brace and (not self._match(TokenType.R_BRACE)):
        self.raise_error('Expecting }')
    return query.order_by(self.expression(exp.Order, expressions=expressions), copy=False)