import numpy as np
from matplotlib import _api
from matplotlib.tri import Triangulation
from matplotlib.tri._trifinder import TriFinder
from matplotlib.tri._tritools import TriAnalyzer
def _cg(A, b, x0=None, tol=1e-10, maxiter=1000):
    """
    Use Preconditioned Conjugate Gradient iteration to solve A x = b
    A simple Jacobi (diagonal) preconditioner is used.

    Parameters
    ----------
    A : _Sparse_Matrix_coo
        *A* must have been compressed before by compress_csc or
        compress_csr method.
    b : array
        Right hand side of the linear system.
    x0 : array, optional
        Starting guess for the solution. Defaults to the zero vector.
    tol : float, optional
        Tolerance to achieve. The algorithm terminates when the relative
        residual is below tol. Default is 1e-10.
    maxiter : int, optional
        Maximum number of iterations.  Iteration will stop after *maxiter*
        steps even if the specified tolerance has not been achieved. Defaults
        to 1000.

    Returns
    -------
    x : array
        The converged solution.
    err : float
        The absolute error np.linalg.norm(A.dot(x) - b)
    """
    n = b.size
    assert A.n == n
    assert A.m == n
    b_norm = np.linalg.norm(b)
    kvec = A.diag
    kvec = np.maximum(kvec, 1e-06)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0
    r = b - A.dot(x)
    w = r / kvec
    p = np.zeros(n)
    beta = 0.0
    rho = np.dot(r, w)
    k = 0
    while np.sqrt(abs(rho)) > tol * b_norm and k < maxiter:
        p = w + beta * p
        z = A.dot(p)
        alpha = rho / np.dot(p, z)
        r = r - alpha * z
        w = r / kvec
        rhoold = rho
        rho = np.dot(r, w)
        x = x + alpha * p
        beta = rho / rhoold
        k += 1
    err = np.linalg.norm(A.dot(x) - b)
    return (x, err)