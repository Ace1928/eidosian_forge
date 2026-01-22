from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_join_parts(self) -> t.Tuple[t.Optional[Token], t.Optional[Token], t.Optional[Token]]:
    is_global = self._match(TokenType.GLOBAL) and self._prev
    kind_pre = self._match_set(self.JOIN_KINDS, advance=False) and self._prev
    if kind_pre:
        kind = self._match_set(self.JOIN_KINDS) and self._prev
        side = self._match_set(self.JOIN_SIDES) and self._prev
        return (is_global, side, kind)
    return (is_global, self._match_set(self.JOIN_SIDES) and self._prev, self._match_set(self.JOIN_KINDS) and self._prev)