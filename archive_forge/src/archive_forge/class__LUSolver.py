from functools import partial
import networkx as nx
from networkx.utils import (
class _LUSolver:
    """LU factorization.

    To solve Ax = b:
        solver = _LUSolver(A)
        x = solver.solve(b)

    optional argument `tol` on solve method is ignored but included
    to match _PCGsolver API.
    """

    def __init__(self, A):
        import scipy as sp
        self._LU = sp.sparse.linalg.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options={'Equil': True, 'SymmetricMode': True})

    def solve(self, B, tol=None):
        import numpy as np
        B = np.asarray(B)
        X = np.ndarray(B.shape, order='F')
        for j in range(B.shape[1]):
            X[:, j] = self._LU.solve(B[:, j])
        return X