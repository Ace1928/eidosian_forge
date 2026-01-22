from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_json_array(self, expr_type: t.Type[E], **kwargs) -> E:
    return self.expression(expr_type, null_handling=self._parse_on_handling('NULL', 'NULL', 'ABSENT'), return_type=self._match_text_seq('RETURNING') and self._parse_type(), strict=self._match_text_seq('STRICT'), **kwargs)