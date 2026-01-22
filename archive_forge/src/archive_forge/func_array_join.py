from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def array_join(col: ColumnOrName, delimiter: str, null_replacement: t.Optional[str]=None) -> Column:
    if null_replacement is not None:
        return Column.invoke_expression_over_column(col, expression.ArrayToString, expression=lit(delimiter), null=lit(null_replacement))
    return Column.invoke_expression_over_column(col, expression.ArrayToString, expression=lit(delimiter))