from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def unalias_group(expression: exp.Expression) -> exp.Expression:
    """
    Replace references to select aliases in GROUP BY clauses.

    Example:
        >>> import sqlglot
        >>> sqlglot.parse_one("SELECT a AS b FROM x GROUP BY b").transform(unalias_group).sql()
        'SELECT a AS b FROM x GROUP BY 1'

    Args:
        expression: the expression that will be transformed.

    Returns:
        The transformed expression.
    """
    if isinstance(expression, exp.Group) and isinstance(expression.parent, exp.Select):
        aliased_selects = {e.alias: i for i, e in enumerate(expression.parent.expressions, start=1) if isinstance(e, exp.Alias)}
        for group_by in expression.expressions:
            if isinstance(group_by, exp.Column) and (not group_by.table) and (group_by.name in aliased_selects):
                group_by.replace(exp.Literal.number(aliased_selects.get(group_by.name)))
    return expression