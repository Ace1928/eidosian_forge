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
def _parse_name_as_expression(self) -> exp.Alias:
    return self.expression(exp.Alias, alias=self._parse_id_var(any_token=True), this=self._match(TokenType.ALIAS) and self._parse_conjunction())