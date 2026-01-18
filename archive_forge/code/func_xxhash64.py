from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def xxhash64(*cols: ColumnOrName) -> Column:
    args = cols[1:] if len(cols) > 1 else []
    return Column.invoke_anonymous_function(cols[0], 'XXHASH64', *args)