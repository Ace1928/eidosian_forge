import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.error import DCPError
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import scalar_value
from cvxpy.utilities import scopes
class Minimize(Objective):
    """An optimization objective for minimization.

    Parameters
    ----------
    expr : Expression
        The expression to minimize. Must be a scalar.

    Raises
    ------
    ValueError
        If expr is not a scalar.
    """
    NAME = 'minimize'

    def __neg__(self) -> 'Maximize':
        return Maximize(-self.args[0])

    def __add__(self, other):
        if not isinstance(other, (Minimize, Maximize)):
            raise NotImplementedError()
        if type(other) is Minimize:
            return Minimize(self.args[0] + other.args[0])
        else:
            raise DCPError('Problem does not follow DCP rules.')

    def canonicalize(self):
        """Pass on the target expression's objective and constraints.
        """
        return self.args[0].canonical_form

    def is_dcp(self, dpp: bool=False) -> bool:
        """The objective must be convex.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_convex()
        return self.args[0].is_convex()

    def is_dgp(self, dpp: bool=False) -> bool:
        """The objective must be log-log convex.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_log_log_convex()
        return self.args[0].is_log_log_convex()

    def is_dpp(self, context='dcp') -> bool:
        with scopes.dpp_scope():
            if context.lower() == 'dcp':
                return self.is_dcp(dpp=True)
            elif context.lower() == 'dgp':
                return self.is_dgp(dpp=True)
            else:
                raise ValueError('Unsupported context ', context)

    def is_dqcp(self) -> bool:
        """The objective must be quasiconvex.
        """
        return self.args[0].is_quasiconvex()

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value.
        """
        return result