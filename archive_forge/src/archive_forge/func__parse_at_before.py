from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _parse_at_before(self, table: exp.Table) -> exp.Table:
    index = self._index
    if self._match_texts(('AT', 'BEFORE')):
        this = self._prev.text.upper()
        kind = self._match(TokenType.L_PAREN) and self._match_texts(self.HISTORICAL_DATA_KIND) and self._prev.text.upper()
        expression = self._match(TokenType.FARROW) and self._parse_bitwise()
        if expression:
            self._match_r_paren()
            when = self.expression(exp.HistoricalData, this=this, kind=kind, expression=expression)
            table.set('when', when)
        else:
            self._retreat(index)
    return table