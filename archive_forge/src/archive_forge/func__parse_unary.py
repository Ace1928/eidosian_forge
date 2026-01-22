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
def _parse_unary(self) -> t.Optional[exp.Expression]:
    if self._match_set(self.UNARY_PARSERS):
        return self.UNARY_PARSERS[self._prev.token_type](self)
    return self._parse_at_time_zone(self._parse_type())