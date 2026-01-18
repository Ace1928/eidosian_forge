from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def zip_with(left: ColumnOrName, right: ColumnOrName, f: t.Callable[[Column, Column], Column]) -> Column:
    f_expression = _get_lambda_from_func(f)
    return Column.invoke_anonymous_function(left, 'ZIP_WITH', right, Column(f_expression))