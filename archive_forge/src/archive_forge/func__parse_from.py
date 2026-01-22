from __future__ import annotations
import typing as t
from sqlglot import exp, parser, tokens
from sqlglot.dialects.dialect import Dialect
from sqlglot.tokens import TokenType
def _parse_from(self, joins: bool=False, skip_from_token: bool=False) -> t.Optional[exp.From]:
    if not skip_from_token and (not self._match(TokenType.FROM)):
        return None
    return self.expression(exp.From, comments=self._prev_comments, this=self._parse_table(joins=joins))