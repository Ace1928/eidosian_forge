from __future__ import annotations
import typing as t
from sqlglot import exp as expression
from sqlglot.dataframe.sql.column import Column
from sqlglot.helper import ensure_list, flatten as _flatten
def _get_lambda_from_func(lambda_expression: t.Callable):
    variables = [expression.to_identifier(x, quoted=_lambda_quoted(x)) for x in lambda_expression.__code__.co_varnames]
    return expression.Lambda(this=lambda_expression(*[Column(x) for x in variables]).expression, expressions=variables)