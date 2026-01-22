import itertools
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import build_scope
def _new_cte(scope, existing_ctes, taken):
    """
    Returns:
        tuple of (name, cte)
        where `name` is a new name for this CTE in the root scope and `cte` is a new CTE instance.
        If this CTE duplicates an existing CTE, `cte` will be None.
    """
    duplicate_cte_alias = existing_ctes.get(scope.expression)
    parent = scope.expression.parent
    name = parent.alias
    if not name:
        name = find_new_name(taken=taken, base='cte')
    if duplicate_cte_alias:
        name = duplicate_cte_alias
    elif taken.get(name):
        name = find_new_name(taken=taken, base=name)
    taken[name] = scope
    if not duplicate_cte_alias:
        existing_ctes[scope.expression] = name
        cte = exp.CTE(this=scope.expression, alias=exp.TableAlias(this=exp.to_identifier(name)))
    else:
        cte = None
    return (name, cte)