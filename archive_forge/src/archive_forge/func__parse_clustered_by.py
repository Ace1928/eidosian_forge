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
def _parse_clustered_by(self) -> exp.ClusteredByProperty:
    self._match_text_seq('BY')
    self._match_l_paren()
    expressions = self._parse_csv(self._parse_column)
    self._match_r_paren()
    if self._match_text_seq('SORTED', 'BY'):
        self._match_l_paren()
        sorted_by = self._parse_csv(self._parse_ordered)
        self._match_r_paren()
    else:
        sorted_by = None
    self._match(TokenType.INTO)
    buckets = self._parse_number()
    self._match_text_seq('BUCKETS')
    return self.expression(exp.ClusteredByProperty, expressions=expressions, sorted_by=sorted_by, buckets=buckets)