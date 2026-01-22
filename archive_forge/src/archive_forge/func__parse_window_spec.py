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
def _parse_window_spec(self) -> t.Dict[str, t.Optional[str | exp.Expression]]:
    self._match(TokenType.BETWEEN)
    return {'value': self._match_text_seq('UNBOUNDED') and 'UNBOUNDED' or (self._match_text_seq('CURRENT', 'ROW') and 'CURRENT ROW') or self._parse_bitwise(), 'side': self._match_texts(self.WINDOW_SIDES) and self._prev.text}