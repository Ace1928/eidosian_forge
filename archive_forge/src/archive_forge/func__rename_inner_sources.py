from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _rename_inner_sources(outer_scope, inner_scope, alias):
    """
    Renames any sources in the inner query that conflict with names in the outer query.

    Args:
        outer_scope (sqlglot.optimizer.scope.Scope)
        inner_scope (sqlglot.optimizer.scope.Scope)
        alias (str)
    """
    taken = set(outer_scope.selected_sources)
    conflicts = taken.intersection(set(inner_scope.selected_sources))
    conflicts -= {alias}
    for conflict in conflicts:
        new_name = find_new_name(taken, conflict)
        source, _ = inner_scope.selected_sources[conflict]
        new_alias = exp.to_identifier(new_name)
        if isinstance(source, exp.Subquery):
            source.set('alias', exp.TableAlias(this=new_alias))
        elif isinstance(source, exp.Table) and source.alias:
            source.set('alias', new_alias)
        elif isinstance(source, exp.Table):
            source.replace(exp.alias_(source, new_alias))
        for column in inner_scope.source_columns(conflict):
            column.set('table', exp.to_identifier(new_name))
        inner_scope.rename_source(conflict, new_name)