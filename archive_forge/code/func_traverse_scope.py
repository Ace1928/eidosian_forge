from __future__ import annotations
import itertools
import logging
import typing as t
from collections import defaultdict
from enum import Enum, auto
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.helper import ensure_collection, find_new_name, seq_get
def traverse_scope(expression: exp.Expression) -> t.List[Scope]:
    """
    Traverse an expression by its "scopes".

    "Scope" represents the current context of a Select statement.

    This is helpful for optimizing queries, where we need more information than
    the expression tree itself. For example, we might care about the source
    names within a subquery. Returns a list because a generator could result in
    incomplete properties which is confusing.

    Examples:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT a FROM (SELECT a FROM x) AS y")
        >>> scopes = traverse_scope(expression)
        >>> scopes[0].expression.sql(), list(scopes[0].sources)
        ('SELECT a FROM x', ['x'])
        >>> scopes[1].expression.sql(), list(scopes[1].sources)
        ('SELECT a FROM (SELECT a FROM x) AS y', ['y'])

    Args:
        expression: Expression to traverse

    Returns:
        A list of the created scope instances
    """
    if isinstance(expression, exp.DDL) and isinstance(expression.expression, exp.Query):
        ddl_with = expression.args.get('with')
        expression = expression.expression
        if ddl_with:
            ddl_with.pop()
            query_ctes = expression.ctes
            if not query_ctes:
                expression.set('with', ddl_with)
            else:
                expression.args['with'].set('recursive', ddl_with.recursive)
                expression.args['with'].set('expressions', [*ddl_with.expressions, *query_ctes])
    if isinstance(expression, exp.Query):
        return list(_traverse_scope(Scope(expression)))
    return []