from typing import Tuple
import numpy as np
import cvxpy.settings as s
from cvxpy.atoms.atom import Atom
class length(Atom):
    """Length of a vector (index of last nonzero, ones-based).
    """

    def __init__(self, x) -> None:
        super(length, self).__init__(x)
        if not self.args[0].is_vector():
            raise ValueError('`length` can only be applied to vectors.')

    @Atom.numpy_numeric
    def numeric(self, values) -> int:
        """Returns the length of x.
        """
        outside_tol = np.abs(values[0]) > s.ATOM_EVAL_TOL
        return np.max(np.nonzero(outside_tol)) + 1

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

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

    def is_atom_quasiconvex(self) -> bool:
        """Is the atom quasiconvex?
        """
        return True

    def is_atom_quasiconcave(self) -> bool:
        """Is the atom quasiconvex?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.args[idx].is_nonpos()

    def _grad(self, values) -> None:
        return None