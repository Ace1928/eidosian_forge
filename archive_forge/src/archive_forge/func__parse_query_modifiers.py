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
def _parse_query_modifiers(self, this: t.Optional[exp.Expression]) -> t.Optional[exp.Expression]:
    if isinstance(this, (exp.Query, exp.Table)):
        for join in self._parse_joins():
            this.append('joins', join)
        for lateral in iter(self._parse_lateral, None):
            this.append('laterals', lateral)
        while True:
            if self._match_set(self.QUERY_MODIFIER_PARSERS, advance=False):
                parser = self.QUERY_MODIFIER_PARSERS[self._curr.token_type]
                key, expression = parser(self)
                if expression:
                    this.set(key, expression)
                    if key == 'limit':
                        offset = expression.args.pop('offset', None)
                        if offset:
                            offset = exp.Offset(expression=offset)
                            this.set('offset', offset)
                            limit_by_expressions = expression.expressions
                            expression.set('expressions', None)
                            offset.set('expressions', limit_by_expressions)
                    continue
            break
    if self.SUPPORTS_IMPLICIT_UNNEST and this and ('from' in this.args):
        this = self._implicit_unnests_to_explicit(this)
    return this