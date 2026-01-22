import warnings
import numpy as np
from cvxpy.constraints.constraint import Constraint
from cvxpy.utilities import scopes
class NonPos(Constraint):
    """An inequality constraint of the form :math:`x \\leq 0`.

    The preferred way of creating an inequality constraint is through
    operator overloading. To constrain an expression ``x`` to be nonpositive,
    write ``x <= 0``; to constrain ``x`` to be nonnegative, write ``x >= 0``.

    Dual variables associated with this constraint are nonnegative, rather
    than nonpositive. As such, dual variables to this constraint belong to the
    polar cone rather than the dual cone.

    Note: strict inequalities are not supported, as they do not make sense in
    a numerical setting.

    Parameters
    ----------
    expr : Expression
        The expression to constrain.
    constr_id : int
        A unique id for the constraint.
    """
    DEPRECATION_MESSAGE = '\n    Explicitly invoking "NonPos(expr)" to a create a constraint is deprecated.\n    Please use operator overloading or "NonNeg(-expr)" instead.\n    \n    Sign conventions on dual variables associated with NonPos constraints may\n    change in the future.\n    '

    def __init__(self, expr, constr_id=None) -> None:
        warnings.warn(NonPos.DEPRECATION_MESSAGE, DeprecationWarning)
        super(NonPos, self).__init__([expr], constr_id)
        if not self.args[0].is_real():
            raise ValueError('Input to NonPos must be real.')

    def name(self) -> str:
        return '%s <= 0' % self.args[0]

    def is_dcp(self, dpp: bool=False) -> bool:
        """A NonPos constraint is DCP if its argument is convex."""
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_convex()
        return self.args[0].is_convex()

    def is_dgp(self, dpp: bool=False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.args[0].is_quasiconvex()

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        ---------
        NumPy.ndarray
        """
        if self.expr.value is None:
            return None
        return np.maximum(self.expr.value, 0)

    def violation(self):
        res = self.residual
        if res is None:
            raise ValueError('Cannot compute the violation of an constraint whose expression is None-valued.')
        viol = np.linalg.norm(res, ord=2)
        return viol