from collections import defaultdict
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name
from sqlglot.optimizer.scope import Scope, traverse_scope
def _is_recursive():
    cte = inner_scope.expression.parent
    node = outer_scope.expression.parent
    while node:
        if node is cte:
            return True
        node = node.parent
    return False