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
def _parse_kill(self) -> exp.Kill:
    kind = exp.var(self._prev.text) if self._match_texts(('CONNECTION', 'QUERY')) else None
    return self.expression(exp.Kill, this=self._parse_primary(), kind=kind)