from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def _string_agg_sql(self: Postgres.Generator, expression: exp.GroupConcat) -> str:
    separator = expression.args.get('separator') or exp.Literal.string(',')
    order = ''
    this = expression.this
    if isinstance(this, exp.Order):
        if this.this:
            this = this.this.pop()
        order = self.sql(expression.this)
    return f'STRING_AGG({self.format_args(this, separator)}{order})'