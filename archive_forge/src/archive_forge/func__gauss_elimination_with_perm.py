from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError
def _gauss_elimination_with_perm(mat, ncols=None, full_elim=False):
    """Gauss elimination of a matrix mat with m rows and n columns.
    If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
    Returns the matrix mat, and the permutation perm that was done on the rows during the process.
    perm[0 : rank] represents the indices of linearly independent rows in the original matrix."""
    mat = np.array(mat, dtype=int, copy=True)
    m = mat.shape[0]
    n = mat.shape[1]
    if ncols is not None:
        n = min(n, ncols)
    perm = np.array(range(m))
    r = 0
    k = 0
    while r < m and k < n:
        is_non_zero = False
        new_r = r
        for j in range(k, n):
            for i in range(r, m):
                if mat[i][j]:
                    is_non_zero = True
                    k = j
                    new_r = i
                    break
            if is_non_zero:
                break
        if not is_non_zero:
            return (mat, perm)
        if new_r != r:
            mat[[r, new_r]] = mat[[new_r, r]]
            perm[r], perm[new_r] = (perm[new_r], perm[r])
        if full_elim:
            for i in range(0, r):
                if mat[i][k]:
                    mat[i] = mat[i] ^ mat[r]
        for i in range(r + 1, m):
            if mat[i][k]:
                mat[i] = mat[i] ^ mat[r]
        r += 1
    return (mat, perm)