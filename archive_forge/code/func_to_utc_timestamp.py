from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def to_utc_timestamp(timestamp: ColumnOrName, tz: ColumnOrName) -> Column:
    tz_column = tz if isinstance(tz, Column) else lit(tz)
    return Column.invoke_expression_over_column(timestamp, expression.FromTimeZone, zone=tz_column)