import operator as op
from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
class NegExpression(UnaryOperator):
    """Negation of an expression.
    """
    OP_NAME = '-'
    OP_FUNC = op.neg

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.args[0].is_nonpos(), self.args[0].is_nonneg())

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return True

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.args[0].is_symmetric()

    def is_hermitian(self) -> bool:
        """Is the expression Hermitian?
        """
        return self.args[0].is_hermitian()

    def graph_implementation(self, arg_objs, shape: Tuple[int, ...], data=None) -> Tuple[lo.LinOp, List[Constraint]]:
        """Negate the affine objective.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.neg_expr(arg_objs[0]), [])