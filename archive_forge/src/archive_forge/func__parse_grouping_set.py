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
def _parse_grouping_set(self) -> t.Optional[exp.Expression]:
    if self._match(TokenType.L_PAREN):
        grouping_set = self._parse_csv(self._parse_column)
        self._match_r_paren()
        return self.expression(exp.Tuple, expressions=grouping_set)
    return self._parse_column()