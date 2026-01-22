from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import flatten, seq_get
from sqlglot.tokens import TokenType
def _sort_array_sql(self: DuckDB.Generator, expression: exp.SortArray) -> str:
    name = 'ARRAY_REVERSE_SORT' if expression.args.get('asc') == exp.false() else 'ARRAY_SORT'
    return self.func(name, expression.this)