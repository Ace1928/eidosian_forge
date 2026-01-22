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
def _parse_window(self, this: t.Optional[exp.Expression], alias: bool=False) -> t.Optional[exp.Expression]:
    func = this
    comments = func.comments if isinstance(func, exp.Expression) else None
    if self._match_pair(TokenType.FILTER, TokenType.L_PAREN):
        self._match(TokenType.WHERE)
        this = self.expression(exp.Filter, this=this, expression=self._parse_where(skip_where_token=True))
        self._match_r_paren()
    if self._match_text_seq('WITHIN', 'GROUP'):
        order = self._parse_wrapped(self._parse_order)
        this = self.expression(exp.WithinGroup, this=this, expression=order)
    if isinstance(this, exp.AggFunc):
        ignore_respect = this.find(exp.IgnoreNulls, exp.RespectNulls)
        if ignore_respect and ignore_respect is not this:
            ignore_respect.replace(ignore_respect.this)
            this = self.expression(ignore_respect.__class__, this=this)
    this = self._parse_respect_or_ignore_nulls(this)
    if alias:
        over = None
        self._match(TokenType.ALIAS)
    elif not self._match_set(self.WINDOW_BEFORE_PAREN_TOKENS):
        return this
    else:
        over = self._prev.text.upper()
    if comments:
        func.comments = None
    if not self._match(TokenType.L_PAREN):
        return self.expression(exp.Window, comments=comments, this=this, alias=self._parse_id_var(False), over=over)
    window_alias = self._parse_id_var(any_token=False, tokens=self.WINDOW_ALIAS_TOKENS)
    first = self._match(TokenType.FIRST)
    if self._match_text_seq('LAST'):
        first = False
    partition, order = self._parse_partition_and_order()
    kind = self._match_set((TokenType.ROWS, TokenType.RANGE)) and self._prev.text
    if kind:
        self._match(TokenType.BETWEEN)
        start = self._parse_window_spec()
        self._match(TokenType.AND)
        end = self._parse_window_spec()
        spec = self.expression(exp.WindowSpec, kind=kind, start=start['value'], start_side=start['side'], end=end['value'], end_side=end['side'])
    else:
        spec = None
    self._match_r_paren()
    window = self.expression(exp.Window, comments=comments, this=this, partition_by=partition, order=order, spec=spec, alias=window_alias, over=over, first=first)
    if self._match_set(self.WINDOW_BEFORE_PAREN_TOKENS, advance=False):
        return self._parse_window(window, alias=alias)
    return window