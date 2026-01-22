from sqlglot import expressions as exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope
def _is_limit_1(scope):
    limit = scope.expression.args.get('limit')
    return limit and limit.expression.this == '1'