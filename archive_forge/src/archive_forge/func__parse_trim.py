from __future__ import annotations
import logging
import typing as t
from collections import defaultdict
from sqlglot import exp
from sqlglot.errors import ErrorLevel, ParseError, concat_messages, merge_errors
from sqlglot.helper import apply_index_offset, ensure_list, seq_get
from sqlglot.time import format_time
from sqlglot.tokens import Token, Tokenizer, TokenType
from sqlglot.trie import TrieResult, in_trie, new_trie
def _parse_trim(self) -> exp.Trim:
    position = None
    collation = None
    expression = None
    if self._match_texts(self.TRIM_TYPES):
        position = self._prev.text.upper()
    this = self._parse_bitwise()
    if self._match_set((TokenType.FROM, TokenType.COMMA)):
        invert_order = self._prev.token_type == TokenType.FROM or self.TRIM_PATTERN_FIRST
        expression = self._parse_bitwise()
        if invert_order:
            this, expression = (expression, this)
    if self._match(TokenType.COLLATE):
        collation = self._parse_bitwise()
    return self.expression(exp.Trim, this=this, position=position, expression=expression, collation=collation)