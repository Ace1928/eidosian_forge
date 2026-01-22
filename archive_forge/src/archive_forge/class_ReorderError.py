from sympy.concrete.expr_with_limits import ExprWithLimits
from sympy.core.singleton import S
from sympy.core.relational import Eq
class ReorderError(NotImplementedError):
    """
    Exception raised when trying to reorder dependent limits.
    """

    def __init__(self, expr, msg):
        super().__init__('%s could not be reordered: %s.' % (expr, msg))