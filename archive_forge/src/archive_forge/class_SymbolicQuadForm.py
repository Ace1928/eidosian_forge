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
class SymbolicQuadForm(Atom):
    """
    Symbolic form of QuadForm when quadratic matrix is not known (yet).
    """

    def __init__(self, x, P, expr) -> None:
        self.original_expression = expr
        super(SymbolicQuadForm, self).__init__(x, P)
        self.P = self.args[1]

    def get_data(self):
        return [self.original_expression]

    def _grad(self, values):
        raise NotImplementedError()

    def is_atom_concave(self) -> bool:
        return self.original_expression.is_atom_concave()

    def is_atom_convex(self) -> bool:
        return self.original_expression.is_atom_convex()

    def is_decr(self, idx) -> bool:
        return self.original_expression.is_decr(idx)

    def is_incr(self, idx) -> bool:
        return self.original_expression.is_incr(idx)

    def shape_from_args(self) -> Tuple[int, ...]:
        return self.original_expression.shape_from_args()

    def sign_from_args(self) -> Tuple[bool, bool]:
        return self.original_expression.sign_from_args()

    def is_quadratic(self) -> bool:
        return True