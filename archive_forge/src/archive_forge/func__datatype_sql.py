from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.parser import binary_range_parser
from sqlglot.tokens import TokenType
def _datatype_sql(self: Postgres.Generator, expression: exp.DataType) -> str:
    if expression.is_type('array'):
        return f'{self.expressions(expression, flat=True)}[]' if expression.expressions else 'ARRAY'
    return self.datatype_sql(expression)