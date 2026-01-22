from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def asc_nulls_first(col: ColumnOrName) -> Column:
    return Column.ensure_col(col).asc_nulls_first()