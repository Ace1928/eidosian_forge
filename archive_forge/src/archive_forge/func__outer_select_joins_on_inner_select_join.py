from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _outer_select_joins_on_inner_select_join():
    """
        All columns from the inner select in the ON clause must be from the first FROM table.

        That is, this can be merged:
            SELECT * FROM x JOIN (SELECT y.a AS a FROM y JOIN z) AS q ON x.a = q.a
                                         ^^^           ^
        But this can't:
            SELECT * FROM x JOIN (SELECT z.a AS a FROM y JOIN z) AS q ON x.a = q.a
                                         ^^^                  ^
        """
    if not isinstance(from_or_join, exp.Join):
        return False
    alias = from_or_join.alias_or_name
    on = from_or_join.args.get('on')
    if not on:
        return False
    selections = [c.name for c in on.find_all(exp.Column) if c.table == alias]
    inner_from = inner_scope.expression.args.get('from')
    if not inner_from:
        return False
    inner_from_table = inner_from.alias_or_name
    inner_projections = {s.alias_or_name: s for s in inner_scope.expression.selects}
    return any((col.table != inner_from_table for selection in selections for col in inner_projections[selection].find_all(exp.Column)))