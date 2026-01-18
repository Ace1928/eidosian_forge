from sqlglot import exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.scope import ScopeType, traverse_scope

    Rewrite sqlglot AST to convert some predicates with subqueries into joins.

    Convert scalar subqueries into cross joins.
    Convert correlated or vectorized subqueries into a group by so it is not a many to many left join.

    Example:
        >>> import sqlglot
        >>> expression = sqlglot.parse_one("SELECT * FROM x AS x WHERE (SELECT y.a AS a FROM y AS y WHERE x.a = y.a) = 1 ")
        >>> unnest_subqueries(expression).sql()
        'SELECT * FROM x AS x LEFT JOIN (SELECT y.a AS a FROM y AS y WHERE TRUE GROUP BY y.a) AS _u_0 ON x.a = _u_0.a WHERE _u_0.a = 1'

    Args:
        expression (sqlglot.Expression): expression to unnest
    Returns:
        sqlglot.Expression: unnested expression
    