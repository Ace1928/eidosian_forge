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
def _parse_with_property(self) -> t.Optional[exp.Expression] | t.List[exp.Expression]:
    if self._match(TokenType.L_PAREN, advance=False):
        return self._parse_wrapped_properties()
    if self._match_text_seq('JOURNAL'):
        return self._parse_withjournaltable()
    if self._match_texts(self.VIEW_ATTRIBUTES):
        return self.expression(exp.ViewAttributeProperty, this=self._prev.text.upper())
    if self._match_text_seq('DATA'):
        return self._parse_withdata(no=False)
    elif self._match_text_seq('NO', 'DATA'):
        return self._parse_withdata(no=True)
    if not self._next:
        return None
    return self._parse_withisolatedloading()