from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_conjunction(self) -> t.Optional[exp.Expression]:
    this = super()._parse_conjunction()
    if self._match(TokenType.PLACEHOLDER):
        return self.expression(exp.If, this=this, true=self._parse_conjunction(), false=self._match(TokenType.COLON) and self._parse_conjunction())
    return this