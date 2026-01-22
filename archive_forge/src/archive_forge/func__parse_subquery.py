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
def _parse_subquery(self, this: t.Optional[exp.Expression], parse_alias: bool=True) -> t.Optional[exp.Subquery]:
    if not this:
        return None
    return self.expression(exp.Subquery, this=this, pivots=self._parse_pivots(), alias=self._parse_table_alias() if parse_alias else None)