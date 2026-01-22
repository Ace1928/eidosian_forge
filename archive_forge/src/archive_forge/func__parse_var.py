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
def _parse_var(self, any_token: bool=False, tokens: t.Optional[t.Collection[TokenType]]=None, upper: bool=False) -> t.Optional[exp.Expression]:
    if any_token and self._advance_any() or self._match(TokenType.VAR) or (self._match_set(tokens) if tokens else False):
        return self.expression(exp.Var, this=self._prev.text.upper() if upper else self._prev.text)
    return self._parse_placeholder()