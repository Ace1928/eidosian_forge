from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def array_remove(col: ColumnOrName, value: ColumnOrLiteral) -> Column:
    value_col = value if isinstance(value, Column) else lit(value)
    return Column.invoke_anonymous_function(col, 'ARRAY_REMOVE', value_col)