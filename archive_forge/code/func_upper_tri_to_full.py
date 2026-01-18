from __future__ import annotations
from typing import Any, Iterable
import scipy.sparse as sp
import cvxpy.lin_ops.lin_utils as lu
from cvxpy import settings as s
from cvxpy.expressions.leaf import Leaf
def upper_tri_to_full(n: int) -> sp.csc_matrix:
    """Returns a coefficient matrix to create a symmetric matrix.

    Parameters
    ----------
    n : int
        The width/height of the matrix.

    Returns
    -------
    SciPy CSC matrix
        The coefficient matrix.
    """
    entries = n * (n + 1) // 2
    val_arr = []
    row_arr = []
    col_arr = []
    count = 0
    for i in range(n):
        for j in range(i, n):
            col_arr.append(count)
            row_arr.append(j * n + i)
            val_arr.append(1.0)
            if i != j:
                col_arr.append(count)
                row_arr.append(i * n + j)
                val_arr.append(1.0)
            count += 1
    return sp.csc_matrix((val_arr, (row_arr, col_arr)), (n * n, entries))