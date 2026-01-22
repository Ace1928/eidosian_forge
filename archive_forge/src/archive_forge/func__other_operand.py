from sqlglot import exp
from sqlglot.helper import name_sequence
from sqlglot.optimizer.scope import ScopeType, traverse_scope
def _other_operand(expression):
    if isinstance(expression, exp.In):
        return expression.this
    if isinstance(expression, (exp.Any, exp.All)):
        return _other_operand(expression.parent)
    if isinstance(expression, exp.Binary):
        return expression.right if isinstance(expression.left, (exp.Subquery, exp.Any, exp.Exists, exp.All)) else expression.left
    return None