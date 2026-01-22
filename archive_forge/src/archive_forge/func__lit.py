from __future__ import annotations
import typing as t
import sqlglot
from sqlglot import expressions as exp
from sqlglot.dataframe.sql.types import DataType
from sqlglot.helper import flatten, is_iterable
@classmethod
def _lit(cls, value: ColumnOrLiteral) -> Column:
    if isinstance(value, dict):
        columns = [cls._lit(v).alias(k).expression for k, v in value.items()]
        return cls(exp.Struct(expressions=columns))
    return cls(exp.convert(value))