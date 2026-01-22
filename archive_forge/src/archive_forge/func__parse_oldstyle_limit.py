from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_oldstyle_limit(self) -> t.Tuple[t.Optional[exp.Expression], t.Optional[exp.Expression]]:
    limit = None
    offset = None
    if self._match_text_seq('LIMIT'):
        parts = self._parse_csv(self._parse_number)
        if len(parts) == 1:
            limit = parts[0]
        elif len(parts) == 2:
            limit = parts[1]
            offset = parts[0]
    return (offset, limit)