import numpy as np
from warnings import warn
from ._optimize import OptimizeResult, OptimizeWarning, _check_unknown_options
from ._linprog_util import _postsolve
def _apply_pivot(T, basis, pivrow, pivcol, tol=1e-09):
    """
    Pivot the simplex tableau inplace on the element given by (pivrow, pivol).
    The entering variable corresponds to the column given by pivcol forcing
    the variable basis[pivrow] to leave the basis.

    Parameters
    ----------
    T : 2-D array
        A 2-D array representing the simplex tableau, T, corresponding to the
        linear programming problem. It should have the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],    0]]

        for a Phase 2 problem, or the form:

        [[A[0, 0], A[0, 1], ..., A[0, n_total], b[0]],
         [A[1, 0], A[1, 1], ..., A[1, n_total], b[1]],
         .
         .
         .
         [A[m, 0], A[m, 1], ..., A[m, n_total], b[m]],
         [c[0],   c[1], ...,   c[n_total],   0],
         [c'[0],  c'[1], ...,  c'[n_total],  0]]

         for a Phase 1 problem (a problem in which a basic feasible solution is
         sought prior to maximizing the actual objective. ``T`` is modified in
         place by ``_solve_simplex``.
    basis : 1-D array
        An array of the indices of the basic variables, such that basis[i]
        contains the column corresponding to the basic variable for row i.
        Basis is modified in place by _apply_pivot.
    pivrow : int
        Row index of the pivot.
    pivcol : int
        Column index of the pivot.
    """
    basis[pivrow] = pivcol
    pivval = T[pivrow, pivcol]
    T[pivrow] = T[pivrow] / pivval
    for irow in range(T.shape[0]):
        if irow != pivrow:
            T[irow] = T[irow] - T[pivrow] * T[irow, pivcol]
    if np.isclose(pivval, tol, atol=0, rtol=10000.0):
        message = f"The pivot operation produces a pivot value of:{pivval: .1e}, which is only slightly greater than the specified tolerance{tol: .1e}. This may lead to issues regarding the numerical stability of the simplex method. Removing redundant constraints, changing the pivot strategy via Bland's rule or increasing the tolerance may help reduce the issue."
        warn(message, OptimizeWarning, stacklevel=5)