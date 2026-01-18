from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def unqualify_unnest(expression: exp.Expression) -> exp.Expression:
    """Remove references to unnest table aliases, added by the optimizer's qualify_columns step."""
    from sqlglot.optimizer.scope import find_all_in_scope
    if isinstance(expression, exp.Select):
        unnest_aliases = {unnest.alias for unnest in find_all_in_scope(expression, exp.Unnest) if isinstance(unnest.parent, (exp.From, exp.Join))}
        if unnest_aliases:
            for column in expression.find_all(exp.Column):
                if column.table in unnest_aliases:
                    column.set('table', None)
                elif column.db in unnest_aliases:
                    column.set('db', None)
    return expression