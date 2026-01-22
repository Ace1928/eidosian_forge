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
def _parse_primary(self) -> t.Optional[exp.Expression]:
    if self._match_set(self.PRIMARY_PARSERS):
        token_type = self._prev.token_type
        primary = self.PRIMARY_PARSERS[token_type](self, self._prev)
        if token_type == TokenType.STRING:
            expressions = [primary]
            while self._match(TokenType.STRING):
                expressions.append(exp.Literal.string(self._prev.text))
            if len(expressions) > 1:
                return self.expression(exp.Concat, expressions=expressions)
        return primary
    if self._match_pair(TokenType.DOT, TokenType.NUMBER):
        return exp.Literal.number(f'0.{self._prev.text}')
    if self._match(TokenType.L_PAREN):
        comments = self._prev_comments
        query = self._parse_select()
        if query:
            expressions = [query]
        else:
            expressions = self._parse_expressions()
        this = self._parse_query_modifiers(seq_get(expressions, 0))
        if isinstance(this, exp.UNWRAPPED_QUERIES):
            this = self._parse_set_operations(self._parse_subquery(this=this, parse_alias=False))
        elif isinstance(this, exp.Subquery):
            this = self._parse_subquery(this=self._parse_set_operations(this), parse_alias=False)
        elif len(expressions) > 1:
            this = self.expression(exp.Tuple, expressions=expressions)
        else:
            this = self.expression(exp.Paren, this=this)
        if this:
            this.add_comments(comments)
        self._match_r_paren(expression=this)
        return this
    return None