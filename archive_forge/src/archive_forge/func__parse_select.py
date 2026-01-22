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
def _parse_select(self, nested: bool=False, table: bool=False, parse_subquery_alias: bool=True, parse_set_operation: bool=True) -> t.Optional[exp.Expression]:
    cte = self._parse_with()
    if cte:
        this = self._parse_statement()
        if not this:
            self.raise_error('Failed to parse any statement following CTE')
            return cte
        if 'with' in this.arg_types:
            this.set('with', cte)
        else:
            self.raise_error(f'{this.key} does not support CTE')
            this = cte
        return this
    from_ = self._parse_from() if self._match(TokenType.FROM, advance=False) else None
    if self._match(TokenType.SELECT):
        comments = self._prev_comments
        hint = self._parse_hint()
        all_ = self._match(TokenType.ALL)
        distinct = self._match_set(self.DISTINCT_TOKENS)
        kind = self._match(TokenType.ALIAS) and self._match_texts(('STRUCT', 'VALUE')) and self._prev.text.upper()
        if distinct:
            distinct = self.expression(exp.Distinct, on=self._parse_value() if self._match(TokenType.ON) else None)
        if all_ and distinct:
            self.raise_error('Cannot specify both ALL and DISTINCT after SELECT')
        limit = self._parse_limit(top=True)
        projections = self._parse_projections()
        this = self.expression(exp.Select, kind=kind, hint=hint, distinct=distinct, expressions=projections, limit=limit)
        this.comments = comments
        into = self._parse_into()
        if into:
            this.set('into', into)
        if not from_:
            from_ = self._parse_from()
        if from_:
            this.set('from', from_)
        this = self._parse_query_modifiers(this)
    elif (table or nested) and self._match(TokenType.L_PAREN):
        if self._match(TokenType.PIVOT):
            this = self._parse_simplified_pivot()
        elif self._match(TokenType.FROM):
            this = exp.select('*').from_(t.cast(exp.From, self._parse_from(skip_from_token=True)))
        else:
            this = self._parse_table() if table else self._parse_select(nested=True, parse_set_operation=False)
            this = self._parse_query_modifiers(self._parse_set_operations(this))
        self._match_r_paren()
        return self._parse_subquery(this, parse_alias=parse_subquery_alias)
    elif self._match(TokenType.VALUES, advance=False):
        this = self._parse_derived_table_values()
    elif from_:
        this = exp.select('*').from_(from_.this, copy=False)
    else:
        this = None
    if parse_set_operation:
        return self._parse_set_operations(this)
    return this