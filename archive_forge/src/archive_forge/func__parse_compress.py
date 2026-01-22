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
def _parse_compress(self) -> exp.CompressColumnConstraint:
    if self._match(TokenType.L_PAREN, advance=False):
        return self.expression(exp.CompressColumnConstraint, this=self._parse_wrapped_csv(self._parse_bitwise))
    return self.expression(exp.CompressColumnConstraint, this=self._parse_bitwise())