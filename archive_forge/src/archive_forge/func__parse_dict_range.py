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
def _parse_dict_range(self, this: str) -> exp.DictRange:
    self._match_l_paren()
    has_min = self._match_text_seq('MIN')
    if has_min:
        min = self._parse_var() or self._parse_primary()
        self._match_text_seq('MAX')
        max = self._parse_var() or self._parse_primary()
    else:
        max = self._parse_var() or self._parse_primary()
        min = exp.Literal.number(0)
    self._match_r_paren()
    return self.expression(exp.DictRange, this=this, min=min, max=max)