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
def _parse_with(self, skip_with_token: bool=False) -> t.Optional[exp.With]:
    if not skip_with_token and (not self._match(TokenType.WITH)):
        return None
    comments = self._prev_comments
    recursive = self._match(TokenType.RECURSIVE)
    expressions = []
    while True:
        expressions.append(self._parse_cte())
        if not self._match(TokenType.COMMA) and (not self._match(TokenType.WITH)):
            break
        else:
            self._match(TokenType.WITH)
    return self.expression(exp.With, comments=comments, expressions=expressions, recursive=recursive)