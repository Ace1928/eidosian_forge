from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def add_within_group_for_percentiles(expression: exp.Expression) -> exp.Expression:
    """Transforms percentiles by adding a WITHIN GROUP clause to them."""
    if isinstance(expression, PERCENTILES) and (not isinstance(expression.parent, exp.WithinGroup)) and expression.expression:
        column = expression.this.pop()
        expression.set('this', expression.expression.pop())
        order = exp.Order(expressions=[exp.Ordered(this=column)])
        expression = exp.WithinGroup(this=expression, expression=order)
    return expression