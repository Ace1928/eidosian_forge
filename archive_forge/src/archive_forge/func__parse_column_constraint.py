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
def _parse_column_constraint(self) -> t.Optional[exp.Expression]:
    if self._match(TokenType.CONSTRAINT):
        this = self._parse_id_var()
    else:
        this = None
    if self._match_texts(self.CONSTRAINT_PARSERS):
        return self.expression(exp.ColumnConstraint, this=this, kind=self.CONSTRAINT_PARSERS[self._prev.text.upper()](self))
    return this