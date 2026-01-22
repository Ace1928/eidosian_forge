import math
import warnings
from itertools import combinations_with_replacement
import cupy as cp
def _build_system(y, d, smoothing, kernel, epsilon, powers):
    """Build the system used to solve for the RBF interpolant coefficients.

    Parameters
    ----------
    y : (P, N) float ndarray
        Data point coordinates.
    d : (P, S) float ndarray
        Data values at `y`.
    smoothing : (P,) float ndarray
        Smoothing parameter for each data point.
    kernel : str
        Name of the RBF.
    epsilon : float
        Shape parameter.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.

    Returns
    -------
    lhs : (P + R, P + R) float ndarray
        Left-hand side matrix.
    rhs : (P + R, S) float ndarray
        Right-hand side matrix.
    shift : (N,) float ndarray
        Domain shift used to create the polynomial matrix.
    scale : (N,) float ndarray
        Domain scaling used to create the polynomial matrix.

    """
    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = NAME_TO_FUNC[kernel]
    mins = cp.min(y, axis=0)
    maxs = cp.max(y, axis=0)
    shift = (maxs + mins) / 2
    scale = (maxs - mins) / 2
    scale[scale == 0.0] = 1.0
    yeps = y * epsilon
    yhat = (y - shift) / scale
    lhs = cp.empty((p + r, p + r), dtype=float).T
    kernel_matrix(yeps, kernel_func, lhs[:p, :p])
    polynomial_matrix(yhat, powers, lhs[:p, p:])
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    for i in range(p):
        lhs[i, i] += smoothing[i]
    rhs = cp.empty((s, p + r), dtype=float).T
    rhs[:p] = d
    rhs[p:] = 0.0
    return (lhs, rhs, shift, scale)