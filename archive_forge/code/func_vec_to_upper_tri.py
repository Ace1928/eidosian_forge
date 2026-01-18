from typing import List, Tuple
import numpy as np
from scipy.sparse import csc_matrix
import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
def vec_to_upper_tri(expr, strict: bool=False):
    """Reshapes a vector into an upper triangular matrix in
    row-major order. The strict argument specifies whether an upper or a strict upper triangular
    matrix should be returned.
    Inverts cp.upper_tri.
    """
    expr = Expression.cast_to_const(expr)
    if not expr.is_vector():
        raise ValueError('The input must be a vector.')
    if expr.ndim != 1:
        expr = vec(expr)
    ell = expr.shape[0]
    if strict:
        n = ((8 * ell + 1) ** 0.5 + 1) // 2
    else:
        n = ((8 * ell + 1) ** 0.5 - 1) // 2
    n = int(n)
    if not (n * (n + 1) // 2 == ell or n * (n - 1) // 2 == ell):
        raise ValueError('The size of the vector must be a triangular number.')
    k = 1 if strict else 0
    row_idx, col_idx = np.triu_indices(n, k=k)
    P_rows = n * row_idx + col_idx
    P_cols = np.arange(ell)
    P_vals = np.ones(P_cols.size)
    P = csc_matrix((P_vals, (P_rows, P_cols)), shape=(n ** 2, ell))
    expr2 = P @ expr
    expr3 = reshape(expr2, (n, n)).T
    return expr3