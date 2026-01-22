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
def _parse_system_versioning_property(self) -> exp.WithSystemVersioningProperty:
    self._match_pair(TokenType.EQ, TokenType.ON)
    prop = self.expression(exp.WithSystemVersioningProperty)
    if self._match(TokenType.L_PAREN):
        self._match_text_seq('HISTORY_TABLE', '=')
        prop.set('this', self._parse_table_parts())
        if self._match(TokenType.COMMA):
            self._match_text_seq('DATA_CONSISTENCY_CHECK', '=')
            prop.set('expression', self._advance_any() and self._prev.text.upper())
        self._match_r_paren()
    return prop