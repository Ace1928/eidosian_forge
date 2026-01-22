from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _parse_query(self) -> t.Optional[exp.Query]:
    from_ = self._parse_from()
    if not from_:
        return None
    query = exp.select('*').from_(from_, copy=False)
    while self._match_texts(self.TRANSFORM_PARSERS):
        query = self.TRANSFORM_PARSERS[self._prev.text.upper()](self, query)
    return query