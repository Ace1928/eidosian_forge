from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_hint(self) -> t.Optional[exp.Hint]:
    if self._match(TokenType.HINT):
        start = self._curr
        while self._curr and (not self._match_pair(TokenType.STAR, TokenType.SLASH)):
            self._advance()
        if not self._curr:
            self.raise_error('Expected */ after HINT')
        end = self._tokens[self._index - 3]
        return exp.Hint(expressions=[self._find_sql(start, end)])
    return None