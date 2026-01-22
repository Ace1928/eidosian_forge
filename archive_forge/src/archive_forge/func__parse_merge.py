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
def _parse_merge(self) -> exp.Merge:
    self._match(TokenType.INTO)
    target = self._parse_table()
    if target and self._match(TokenType.ALIAS, advance=False):
        target.set('alias', self._parse_table_alias())
    self._match(TokenType.USING)
    using = self._parse_table()
    self._match(TokenType.ON)
    on = self._parse_conjunction()
    return self.expression(exp.Merge, this=target, using=using, on=on, expressions=self._parse_when_matched())