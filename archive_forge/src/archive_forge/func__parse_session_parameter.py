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
def _parse_session_parameter(self) -> exp.SessionParameter:
    kind = None
    this = self._parse_id_var() or self._parse_primary()
    if this and self._match(TokenType.DOT):
        kind = this.name
        this = self._parse_var() or self._parse_primary()
    return self.expression(exp.SessionParameter, this=this, kind=kind)