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
def _match_pair(self, token_type_a, token_type_b, advance=True):
    if not self._curr or not self._next:
        return None
    if self._curr.token_type == token_type_a and self._next.token_type == token_type_b:
        if advance:
            self._advance(2)
        return True
    return None