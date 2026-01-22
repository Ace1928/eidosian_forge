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
def _parse_property_assignment(self, exp_class: t.Type[E], **kwargs: t.Any) -> E:
    self._match(TokenType.EQ)
    self._match(TokenType.ALIAS)
    field = self._parse_field()
    if isinstance(field, exp.Identifier) and (not field.quoted):
        field = exp.var(field)
    return self.expression(exp_class, this=field, **kwargs)