from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _merge_where(outer_scope, inner_scope, from_or_join):
    """
    Merge WHERE clause of inner query into outer query.

    Args:
        outer_scope (sqlglot.optimizer.scope.Scope)
        inner_scope (sqlglot.optimizer.scope.Scope)
        from_or_join (exp.From|exp.Join)
    """
    where = inner_scope.expression.args.get('where')
    if not where or not where.this:
        return
    expression = outer_scope.expression
    if isinstance(from_or_join, exp.Join):
        from_ = expression.args.get('from')
        sources = {from_.alias_or_name} if from_ else {}
        for join in expression.args['joins']:
            source = join.alias_or_name
            sources.add(source)
            if source == from_or_join.alias_or_name:
                break
        if exp.column_table_names(where.this) <= sources:
            from_or_join.on(where.this, copy=False)
            from_or_join.set('on', from_or_join.args.get('on'))
            return
    expression.where(where.this, copy=False)