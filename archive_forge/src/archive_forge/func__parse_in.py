from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import is_int, seq_get
from sqlglot.tokens import Token, TokenType
def _parse_in(self, this: t.Optional[exp.Expression], is_global: bool=False) -> exp.In:
    this = super()._parse_in(this)
    this.set('is_global', is_global)
    return this