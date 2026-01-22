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
def _parse_bitwise(self) -> t.Optional[exp.Expression]:
    this = self._parse_term()
    while True:
        if self._match_set(self.BITWISE):
            this = self.expression(self.BITWISE[self._prev.token_type], this=this, expression=self._parse_term())
        elif self.dialect.DPIPE_IS_STRING_CONCAT and self._match(TokenType.DPIPE):
            this = self.expression(exp.DPipe, this=this, expression=self._parse_term(), safe=not self.dialect.STRICT_STRING_CONCAT)
        elif self._match(TokenType.DQMARK):
            this = self.expression(exp.Coalesce, this=this, expressions=self._parse_term())
        elif self._match_pair(TokenType.LT, TokenType.LT):
            this = self.expression(exp.BitwiseLeftShift, this=this, expression=self._parse_term())
        elif self._match_pair(TokenType.GT, TokenType.GT):
            this = self.expression(exp.BitwiseRightShift, this=this, expression=self._parse_term())
        else:
            break
    return this