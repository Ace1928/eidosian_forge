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
def _parse_when_matched(self) -> t.List[exp.When]:
    whens = []
    while self._match(TokenType.WHEN):
        matched = not self._match(TokenType.NOT)
        self._match_text_seq('MATCHED')
        source = False if self._match_text_seq('BY', 'TARGET') else self._match_text_seq('BY', 'SOURCE')
        condition = self._parse_conjunction() if self._match(TokenType.AND) else None
        self._match(TokenType.THEN)
        if self._match(TokenType.INSERT):
            _this = self._parse_star()
            if _this:
                then: t.Optional[exp.Expression] = self.expression(exp.Insert, this=_this)
            else:
                then = self.expression(exp.Insert, this=self._parse_value(), expression=self._match_text_seq('VALUES') and self._parse_value())
        elif self._match(TokenType.UPDATE):
            expressions = self._parse_star()
            if expressions:
                then = self.expression(exp.Update, expressions=expressions)
            else:
                then = self.expression(exp.Update, expressions=self._match(TokenType.SET) and self._parse_csv(self._parse_equality))
        elif self._match(TokenType.DELETE):
            then = self.expression(exp.Var, this=self._prev.text)
        else:
            then = None
        whens.append(self.expression(exp.When, matched=matched, source=source, condition=condition, then=then))
    return whens