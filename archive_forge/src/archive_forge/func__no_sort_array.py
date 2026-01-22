from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _no_sort_array(self: Presto.Generator, expression: exp.SortArray) -> str:
    if expression.args.get('asc') == exp.false():
        comparator = '(a, b) -> CASE WHEN a < b THEN 1 WHEN a > b THEN -1 ELSE 0 END'
    else:
        comparator = None
    return self.func('ARRAY_SORT', expression.this, comparator)