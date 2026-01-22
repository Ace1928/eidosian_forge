import numpy as np
from scipy.linalg import svd
from scipy.linalg.interpolative import interp_decomp
import scipy
from scipy.linalg.blas import dtrsm
def _remove_redundancy_svd(A, b):
    """
    Eliminates redundant equations from system of equations defined by Ax = b
    and identifies infeasibilities.

    Parameters
    ----------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations

    Returns
    -------
    A : 2-D array
        An array representing the left-hand side of a system of equations
    b : 1-D array
        An array representing the right-hand side of a system of equations
    status: int
        An integer indicating the status of the system
        0: No infeasibility identified
        2: Trivially infeasible
    message : str
        A string descriptor of the exit status of the optimization.

    References
    ----------
    .. [2] Andersen, Erling D. "Finding all linearly dependent rows in
           large-scale linear programming." Optimization Methods and Software
           6.3 (1995): 219-227.

    """
    A, b, status, message = _remove_zero_rows(A, b)
    if status != 0:
        return (A, b, status, message)
    U, s, Vh = svd(A)
    eps = np.finfo(float).eps
    tol = s.max() * max(A.shape) * eps
    m, n = A.shape
    s_min = s[-1] if m <= n else 0
    while abs(s_min) < tol:
        v = U[:, -1]
        eligibleRows = np.abs(v) > tol * 10000000.0
        if not np.any(eligibleRows) or np.any(np.abs(v.dot(A)) > tol):
            status = 4
            message = 'Due to numerical issues, redundant equality constraints could not be removed automatically. Try providing your constraint matrices as sparse matrices to activate sparse presolve, try turning off redundancy removal, or try turning off presolve altogether.'
            break
        if np.any(np.abs(v.dot(b)) > tol * 100):
            status = 2
            message = 'There is a linear combination of rows of A_eq that results in zero, suggesting a redundant constraint. However the same linear combination of b_eq is nonzero, suggesting that the constraints conflict and the problem is infeasible.'
            break
        i_remove = _get_densest(A, eligibleRows)
        A = np.delete(A, i_remove, axis=0)
        b = np.delete(b, i_remove)
        U, s, Vh = svd(A)
        m, n = A.shape
        s_min = s[-1] if m <= n else 0
    return (A, b, status, message)