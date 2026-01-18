from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np
import scipy as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
def vectorized_lower_tri_to_triples(A: sp.sparse.coo_matrix | list[float] | np.ndarray, dim: int) -> tuple[list[int], list[int], list[float]]:
    """
    Attributes
    ----------
    A : scipy.sparse.coo_matrix | list[float] | np.ndarray
        Contains the lower triangular entries of a symmetric matrix, flattened into a 1D array in
        column-major order.
    dim : int
        The number of rows (equivalently, columns) in the original matrix.

    Returns
    -------
    rows : list[int]
        The row indices of the entries in the original matrix.
    cols : list[int]
        The column indices of the entries in the original matrix.
    vals : list[float]
        The values of the entries in the original matrix.
    """
    if isinstance(A, sp.sparse.coo_matrix):
        vals = A.data
        flattened_cols = A.col
        if not np.all(flattened_cols[:-1] < flattened_cols[1:]):
            sort_idx = np.argsort(flattened_cols)
            vals = vals[sort_idx]
            flattened_cols = flattened_cols[sort_idx]
    elif isinstance(A, list):
        vals = A
        flattened_cols = np.arange(len(A))
    elif isinstance(A, np.ndarray):
        vals = list(A)
        flattened_cols = np.arange(len(A))
    else:
        raise TypeError(f'Expected A to be a coo_matrix, list, or ndarray, but got {type(A)} instead.')
    cum_cols = np.cumsum(np.arange(dim, 0, -1))
    rows, cols = ([], [])
    current_col = 0
    for v in flattened_cols:
        for c in range(current_col, dim):
            if v < cum_cols[c]:
                cols.append(c)
                prev_row = 0 if c == 0 else cum_cols[c - 1]
                rows.append(v - prev_row + c)
                break
            else:
                current_col += 1
    return (rows, cols, vals)