from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _merge_joins(outer_scope, inner_scope, from_or_join):
    """
    Merge JOIN clauses of inner query into outer query.

    Args:
        outer_scope (sqlglot.optimizer.scope.Scope)
        inner_scope (sqlglot.optimizer.scope.Scope)
        from_or_join (exp.From|exp.Join)
    """
    new_joins = []
    joins = inner_scope.expression.args.get('joins') or []
    for join in joins:
        new_joins.append(join)
        outer_scope.add_source(join.alias_or_name, inner_scope.sources[join.alias_or_name])
    if new_joins:
        outer_joins = outer_scope.expression.args.get('joins', [])
        if isinstance(from_or_join, exp.From):
            position = 0
        else:
            position = outer_joins.index(from_or_join) + 1
        outer_joins[position:position] = new_joins
        outer_scope.expression.set('joins', outer_joins)