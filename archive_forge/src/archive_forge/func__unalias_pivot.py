from __future__ import annotations
import typing as t
from sqlglot import exp, transforms
from sqlglot.dialects.dialect import (
from sqlglot.dialects.hive import Hive
from sqlglot.helper import seq_get
from sqlglot.transforms import (
def _unalias_pivot(expression: exp.Expression) -> exp.Expression:
    """
    Spark doesn't allow PIVOT aliases, so we need to remove them and possibly wrap a
    pivoted source in a subquery with the same alias to preserve the query's semantics.

    Example:
        >>> from sqlglot import parse_one
        >>> expr = parse_one("SELECT piv.x FROM tbl PIVOT (SUM(a) FOR b IN ('x')) piv")
        >>> print(_unalias_pivot(expr).sql(dialect="spark"))
        SELECT piv.x FROM (SELECT * FROM tbl PIVOT(SUM(a) FOR b IN ('x'))) AS piv
    """
    if isinstance(expression, exp.From) and expression.this.args.get('pivots'):
        pivot = expression.this.args['pivots'][0]
        if pivot.alias:
            alias = pivot.args['alias'].pop()
            return exp.From(this=expression.this.replace(exp.select('*').from_(expression.this.copy(), copy=False).subquery(alias=alias, copy=False)))
    return expression