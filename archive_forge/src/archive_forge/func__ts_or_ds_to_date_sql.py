from __future__ import annotations
import typing as t
from sqlglot import exp, generator, parser, tokens, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.dialects.mysql import MySQL
from sqlglot.helper import apply_index_offset, seq_get
from sqlglot.tokens import TokenType
def _ts_or_ds_to_date_sql(self: Presto.Generator, expression: exp.TsOrDsToDate) -> str:
    time_format = self.format_time(expression)
    if time_format and time_format not in (Presto.TIME_FORMAT, Presto.DATE_FORMAT):
        return self.sql(exp.cast(_str_to_time_sql(self, expression), exp.DataType.Type.DATE))
    return self.sql(exp.cast(exp.cast(expression.this, exp.DataType.Type.TIMESTAMP), exp.DataType.Type.DATE))