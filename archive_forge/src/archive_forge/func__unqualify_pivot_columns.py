from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.helper import seq_get
from sqlglot.transforms import (
def _unqualify_pivot_columns(expression: exp.Expression) -> exp.Expression:
    """
    Spark doesn't allow the column referenced in the PIVOT's field to be qualified,
    so we need to unqualify it.

    Example:
        >>> from sqlglot import parse_one
        >>> expr = parse_one("SELECT * FROM tbl PIVOT (SUM(tbl.sales) FOR tbl.quarter IN ('Q1', 'Q2'))")
        >>> print(_unqualify_pivot_columns(expr).sql(dialect="spark"))
        SELECT * FROM tbl PIVOT(SUM(tbl.sales) FOR quarter IN ('Q1', 'Q1'))
    """
    if isinstance(expression, exp.Pivot):
        expression.set('field', transforms.unqualify_columns(expression.args['field']))
    return expression