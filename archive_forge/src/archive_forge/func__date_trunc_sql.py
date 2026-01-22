from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get
from sqlglot.tokens import TokenType
def _date_trunc_sql(self: MySQL.Generator, expression: exp.DateTrunc) -> str:
    expr = self.sql(expression, 'this')
    unit = expression.text('unit').upper()
    if unit == 'WEEK':
        concat = f"CONCAT(YEAR({expr}), ' ', WEEK({expr}, 1), ' 1')"
        date_format = '%Y %u %w'
    elif unit == 'MONTH':
        concat = f"CONCAT(YEAR({expr}), ' ', MONTH({expr}), ' 1')"
        date_format = '%Y %c %e'
    elif unit == 'QUARTER':
        concat = f"CONCAT(YEAR({expr}), ' ', QUARTER({expr}) * 3 - 2, ' 1')"
        date_format = '%Y %c %e'
    elif unit == 'YEAR':
        concat = f"CONCAT(YEAR({expr}), ' 1 1')"
        date_format = '%Y %c %e'
    else:
        if unit != 'DAY':
            self.unsupported(f'Unexpected interval unit: {unit}')
        return self.func('DATE', expr)
    return self.func('STR_TO_DATE', concat, f"'{date_format}'")