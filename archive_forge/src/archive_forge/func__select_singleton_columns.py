import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult
def _select_singleton_columns(A, b):
    """
    Finds singleton columns for which the singleton entry is of the same sign
    as the right-hand side; these columns are eligible for inclusion in an
    initial basis. Determines the rows in which the singleton entries are
    located. For each of these rows, returns the indices of the one singleton
    column and its corresponding row.
    """
    column_indices = np.nonzero(np.sum(np.abs(A) != 0, axis=0) == 1)[0]
    columns = A[:, column_indices]
    row_indices = np.zeros(len(column_indices), dtype=int)
    nonzero_rows, nonzero_columns = np.nonzero(columns)
    row_indices[nonzero_columns] = nonzero_rows
    same_sign = A[row_indices, column_indices] * b[row_indices] >= 0
    column_indices = column_indices[same_sign][::-1]
    row_indices = row_indices[same_sign][::-1]
    unique_row_indices, first_columns = np.unique(row_indices, return_index=True)
    return (column_indices[first_columns], unique_row_indices)