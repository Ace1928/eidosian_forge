from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, is_float, is_int, seq_get
from sqlglot.tokens import TokenType
def _parse_bracket_key_value(self, is_map: bool=False) -> t.Optional[exp.Expression]:
    if is_map:
        return self._parse_slice(self._parse_string())
    return self._parse_slice(self._parse_alias(self._parse_conjunction(), explicit=True))