from sqlglot import expressions as exp
from sqlglot.optimizer.normalize import normalized
from sqlglot.optimizer.scope import Scope, traverse_scope
def _has_single_output_row(scope):
    return isinstance(scope.expression, exp.Select) and (all((isinstance(e.unalias(), exp.AggFunc) for e in scope.expression.selects)) or _is_limit_1(scope) or (not scope.expression.args.get('from')))