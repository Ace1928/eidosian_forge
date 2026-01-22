import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import solve
from ._optimize import _check_unknown_options
from ._bglu_dense import LU
from ._bglu_dense import BGLU as BGLU
from ._linprog_util import _postsolve
from ._optimize import OptimizeResult
def _generate_auxiliary_problem(A, b, x0, tol):
    """
    Modifies original problem to create an auxiliary problem with a trivial
    initial basic feasible solution and an objective that minimizes
    infeasibility in the original problem.

    Conceptually, this is done by stacking an identity matrix on the right of
    the original constraint matrix, adding artificial variables to correspond
    with each of these new columns, and generating a cost vector that is all
    zeros except for ones corresponding with each of the new variables.

    A initial basic feasible solution is trivial: all variables are zero
    except for the artificial variables, which are set equal to the
    corresponding element of the right hand side `b`.

    Running the simplex method on this auxiliary problem drives all of the
    artificial variables - and thus the cost - to zero if the original problem
    is feasible. The original problem is declared infeasible otherwise.

    Much of the complexity below is to improve efficiency by using singleton
    columns in the original problem where possible, thus generating artificial
    variables only as necessary, and using an initial 'guess' basic feasible
    solution.
    """
    status = 0
    m, n = A.shape
    if x0 is not None:
        x = x0
    else:
        x = np.zeros(n)
    r = b - A @ x
    A[r < 0] = -A[r < 0]
    b[r < 0] = -b[r < 0]
    r[r < 0] *= -1
    if x0 is None:
        nonzero_constraints = np.arange(m)
    else:
        nonzero_constraints = np.where(r > tol)[0]
    basis = np.where(np.abs(x) > tol)[0]
    if len(nonzero_constraints) == 0 and len(basis) <= m:
        c = np.zeros(n)
        basis = _get_more_basis_columns(A, basis)
        return (A, b, c, basis, x, status)
    elif len(nonzero_constraints) > m - len(basis) or np.any(x < 0):
        c = np.zeros(n)
        status = 6
        return (A, b, c, basis, x, status)
    cols, rows = _select_singleton_columns(A, r)
    i_tofix = np.isin(rows, nonzero_constraints)
    i_notinbasis = np.logical_not(np.isin(cols, basis))
    i_fix_without_aux = np.logical_and(i_tofix, i_notinbasis)
    rows = rows[i_fix_without_aux]
    cols = cols[i_fix_without_aux]
    arows = nonzero_constraints[np.logical_not(np.isin(nonzero_constraints, rows))]
    n_aux = len(arows)
    acols = n + np.arange(n_aux)
    basis_ng = np.concatenate((cols, acols))
    basis_ng_rows = np.concatenate((rows, arows))
    A = np.hstack((A, np.zeros((m, n_aux))))
    A[arows, acols] = 1
    x = np.concatenate((x, np.zeros(n_aux)))
    x[basis_ng] = r[basis_ng_rows] / A[basis_ng_rows, basis_ng]
    c = np.zeros(n_aux + n)
    c[acols] = 1
    basis = np.concatenate((basis, basis_ng))
    basis = _get_more_basis_columns(A, basis)
    return (A, b, c, basis, x, status)