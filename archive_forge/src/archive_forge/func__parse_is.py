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
def _parse_is(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    index = self._index - 1
    negate = self._match(TokenType.NOT)
    if self._match_text_seq('DISTINCT', 'FROM'):
        klass = exp.NullSafeEQ if negate else exp.NullSafeNEQ
        return self.expression(klass, this=this, expression=self._parse_bitwise())
    expression = self._parse_null() or self._parse_boolean()
    if not expression:
        self._retreat(index)
        return None
    this = self.expression(exp.Is, this=this, expression=expression)
    return self.expression(exp.Not, this=this) if negate else this