from __future__ import annotations
import logging
import re
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.helper import seq_get, split_num_words
from sqlglot.tokens import TokenType
def _ts_or_ds_diff_sql(self: BigQuery.Generator, expression: exp.TsOrDsDiff) -> str:
    expression.this.replace(exp.cast(expression.this, exp.DataType.Type.TIMESTAMP))
    expression.expression.replace(exp.cast(expression.expression, exp.DataType.Type.TIMESTAMP))
    unit = unit_to_var(expression)
    return self.func('DATE_DIFF', expression.this, expression.expression, unit)