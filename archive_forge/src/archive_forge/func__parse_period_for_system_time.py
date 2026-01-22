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
def _parse_period_for_system_time(self) -> t.Optional[exp.PeriodForSystemTimeConstraint]:
    if not self._match(TokenType.TIMESTAMP_SNAPSHOT):
        self._retreat(self._index - 1)
        return None
    id_vars = self._parse_wrapped_id_vars()
    return self.expression(exp.PeriodForSystemTimeConstraint, this=seq_get(id_vars, 0), expression=seq_get(id_vars, 1))