from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _parse_for_in(self) -> exp.ForIn:
    this = self._parse_range()
    self._match_text_seq('DO')
    return self.expression(exp.ForIn, this=this, expression=self._parse_statement())