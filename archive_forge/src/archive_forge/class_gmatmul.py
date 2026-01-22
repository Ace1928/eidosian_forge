from typing import Tuple
import numpy as np
import cvxpy.utilities as u
from cvxpy.atoms.atom import Atom
from cvxpy.expressions import cvxtypes
class gmatmul(Atom):
    """Geometric matrix multiplication; :math:`A \\mathbin{\\diamond} X`.

    For :math:`A \\in \\mathbf{R}^{m \\times n}` and
    :math:`X \\in \\mathbf{R}^{n \\times p}_{++}`, this atom represents

    .. math::

        \\left[\\begin{array}{ccc}
         \\prod_{j=1}^n X_{j1}^{A_{1j}} & \\cdots & \\prod_{j=1}^n X_{pj}^{A_{1j}} \\\\
         \\vdots &  & \\vdots \\\\
         \\prod_{j=1}^n X_{j1}^{A_{mj}} & \\cdots & \\prod_{j=1}^n X_{pj}^{A_{mj}}
        \\end{array}\\right]

    This atom is log-log affine (in :math:`X`).

    Parameters
    ----------
    A : cvxpy.Expression
        A constant matrix.
    X : cvxpy.Expression
        A positive matrix.
    """

    def __init__(self, A, X) -> None:
        self.A = Atom.cast_to_const(A)
        super(gmatmul, self).__init__(X)

    def numeric(self, values):
        """Geometric matrix multiplication.
        """
        logX = np.log(values[0])
        return np.exp(self.A.value @ logX)

    def name(self) -> str:
        return '%s(%s, %s)' % (self.__class__.__name__, self.A, self.args[0])

    def validate_arguments(self) -> None:
        """Raises an error if the arguments are invalid.
        """
        super(gmatmul, self).validate_arguments()
        if not self.A.is_constant():
            raise ValueError('gmatmul(A, X) requires that A be constant.')
        if self.A.parameters() and (not isinstance(self.A, cvxtypes.parameter())):
            raise ValueError('gmatmul(A, X) requires that A be a Constant or a Parameter.')
        if not self.args[0].is_pos():
            raise ValueError('gmatmul(A, X) requires that X be positive.')

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return u.shape.mul_shapes(self.A.shape, self.args[0].shape)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.A]

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def parameters(self):
        return self.args[0].parameters() + self.A.parameters()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        if u.scopes.dpp_scope_active():
            X = self.args[0]
            A = self.A
            return not (X.parameters() and A.parameters())
        else:
            return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return self.is_atom_log_log_convex()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.A.is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.A.is_nonpos()

    def _grad(self, values) -> None:
        return None