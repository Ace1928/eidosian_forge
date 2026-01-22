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
def _parse_comprehension(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Comprehension]:
    index = self._index
    expression = self._parse_column()
    if not self._match(TokenType.IN):
        self._retreat(index - 1)
        return None
    iterator = self._parse_column()
    condition = self._parse_conjunction() if self._match_text_seq('IF') else None
    return self.expression(exp.Comprehension, this=this, expression=expression, iterator=iterator, condition=condition)