from __future__ import division
import warnings
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from scipy import linalg as LA
from cvxpy.atoms.affine.wraps import psd_wrap
from cvxpy.atoms.atom import Atom
from cvxpy.expressions.expression import Expression
from cvxpy.interface.matrix_utilities import is_sparse
from cvxpy.utilities.linalg import sparse_cholesky
class QuadForm(Atom):
    _allow_complex = True

    def __init__(self, x, P) -> None:
        """Atom representing :math:`x^T P x`."""
        super(QuadForm, self).__init__(x, P)

    def numeric(self, values):
        prod = values[1].dot(values[0])
        if self.args[0].is_complex():
            quad = np.dot(np.conj(values[0]).T, prod)
        else:
            quad = np.dot(np.transpose(values[0]), prod)
        return np.real(quad)

    def validate_arguments(self) -> None:
        super(QuadForm, self).validate_arguments()
        n = self.args[1].shape[0]
        if self.args[1].shape[1] != n or self.args[0].shape not in [(n, 1), (n,)]:
            raise ValueError('Invalid dimensions for arguments.')
        if not self.args[1].is_hermitian():
            raise ValueError('Quadratic form matrices must be symmetric/Hermitian.')

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.is_atom_convex(), self.is_atom_concave())

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        P = self.args[1]
        return P.is_constant() and P.is_psd()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        P = self.args[1]
        return P.is_constant() and P.is_nsd()

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
        return self.args[0].is_nonneg() and self.args[1].is_nonneg() or (self.args[0].is_nonpos() and self.args[1].is_nonneg())

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_nonneg() and self.args[1].is_nonpos() or (self.args[0].is_nonpos() and self.args[1].is_nonpos())

    def is_quadratic(self) -> bool:
        """Is the atom quadratic?
        """
        return True

    def has_quadratic_term(self) -> bool:
        """Always a quadratic term.
        """
        return True

    def is_pwl(self) -> bool:
        """Is the atom piecewise linear?
        """
        return False

    def name(self) -> str:
        return '%s(%s, %s)' % (self.__class__.__name__, self.args[0], self.args[1])

    def _grad(self, values):
        x = np.array(values[0])
        P = np.array(values[1])
        D = (P + np.conj(P.T)) @ x
        return [sp.csc_matrix(D.ravel(order='F')).T]

    def shape_from_args(self) -> Tuple[int, ...]:
        return tuple()