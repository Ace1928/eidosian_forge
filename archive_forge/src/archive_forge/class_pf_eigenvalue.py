from typing import Tuple
import numpy as np
from cvxpy.atoms.atom import Atom
class pf_eigenvalue(Atom):
    """The Perron-Frobenius eigenvalue of a positive matrix.

    For an elementwise positive matrix :math:`X`, this atom represents its
    spectral radius, i.e., the magnitude of its largest eigenvalue. Because
    :math:`X` is positive, the spectral radius equals its largest eigenvalue,
    which is guaranteed to be positive.

    This atom is log-log convex.

    Parameters
    ----------
    X : cvxpy.Expression
        A positive square matrix.
    """

    def __init__(self, X) -> None:
        super(pf_eigenvalue, self).__init__(X)
        if len(X.shape) != 2 or X.shape[0] != X.shape[1]:
            raise ValueError('Argument to `spectral radius` must be a square matrix, received ', X)
        self.args[0] = X

    def numeric(self, values):
        return np.max(np.abs(np.linalg.eig(values[0])[0]))

    def name(self) -> str:
        return '%s(%s)' % (self.__class__.__name__, self.args[0])

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

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return True

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def _grad(self, values) -> None:
        return None