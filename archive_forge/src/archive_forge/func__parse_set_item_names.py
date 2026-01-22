from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _parse_set_item_names(self) -> exp.Expression:
    charset = self._parse_string() or self._parse_id_var()
    if self._match_text_seq('COLLATE'):
        collate = self._parse_string() or self._parse_id_var()
    else:
        collate = None
    return self.expression(exp.SetItem, this=charset, collate=collate, kind='NAMES')