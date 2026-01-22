import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def _woodbury_algorithm(A, ur, ll, b, k):
    """
    Solve a cyclic banded linear system with upper right
    and lower blocks of size ``(k-1) / 2`` using
    the Woodbury formula

    Parameters
    ----------
    A : 2-D array, shape(k, n)
        Matrix of diagonals of original matrix (see
        ``solve_banded`` documentation).
    ur : 2-D array, shape(bs, bs)
        Upper right block matrix.
    ll : 2-D array, shape(bs, bs)
        Lower left block matrix.
    b : 1-D array, shape(n,)
        Vector of constant terms of the system of linear equations.
    k : int
        B-spline degree.

    Returns
    -------
    c : 1-D array, shape(n,)
        Solution of the original system of linear equations.

    Notes
    -----
    This algorithm works only for systems with banded matrix A plus
    a correction term U @ V.T, where the matrix U @ V.T gives upper right
    and lower left block of A
    The system is solved with the following steps:
        1.  New systems of linear equations are constructed:
            A @ z_i = u_i,
            u_i - column vector of U,
            i = 1, ..., k - 1
        2.  Matrix Z is formed from vectors z_i:
            Z = [ z_1 | z_2 | ... | z_{k - 1} ]
        3.  Matrix H = (1 + V.T @ Z)^{-1}
        4.  The system A' @ y = b is solved
        5.  x = y - Z @ (H @ V.T @ y)
    Also, ``n`` should be greater than ``k``, otherwise corner block
    elements will intersect with diagonals.

    Examples
    --------
    Consider the case of n = 8, k = 5 (size of blocks - 2 x 2).
    The matrix of a system:       U:          V:
      x  x  x  *  *  a  b         a b 0 0     0 0 1 0
      x  x  x  x  *  *  c         0 c 0 0     0 0 0 1
      x  x  x  x  x  *  *         0 0 0 0     0 0 0 0
      *  x  x  x  x  x  *         0 0 0 0     0 0 0 0
      *  *  x  x  x  x  x         0 0 0 0     0 0 0 0
      d  *  *  x  x  x  x         0 0 d 0     1 0 0 0
      e  f  *  *  x  x  x         0 0 e f     0 1 0 0

    References
    ----------
    .. [1] William H. Press, Saul A. Teukolsky, William T. Vetterling
           and Brian P. Flannery, Numerical Recipes, 2007, Section 2.7.3

    """
    k_mod = k - k % 2
    bs = int((k - 1) / 2) + (k + 1) % 2
    n = A.shape[1] + 1
    U = np.zeros((n - 1, k_mod))
    VT = np.zeros((k_mod, n - 1))
    U[:bs, :bs] = ur
    VT[np.arange(bs), np.arange(bs) - bs] = 1
    U[-bs:, -bs:] = ll
    VT[np.arange(bs) - bs, np.arange(bs)] = 1
    Z = solve_banded((bs, bs), A, U)
    H = solve(np.identity(k_mod) + VT @ Z, np.identity(k_mod))
    y = solve_banded((bs, bs), A, b)
    c = y - Z @ (H @ (VT @ y))
    return c