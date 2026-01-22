import functools
import numpy as np
from numpy.linalg import norm
from scipy.sparse.linalg import LinearOperator
from ..sparse import issparse, csc_matrix, csr_matrix, coo_matrix, find
from ._group_columns import group_dense, group_sparse
from scipy._lib._array_api import atleast_nd, array_namespace
def _prepare_bounds(bounds, x0):
    """
    Prepares new-style bounds from a two-tuple specifying the lower and upper
    limits for values in x0. If a value is not bound then the lower/upper bound
    will be expected to be -np.inf/np.inf.

    Examples
    --------
    >>> _prepare_bounds([(0, 1, 2), (1, 2, np.inf)], [0.5, 1.5, 2.5])
    (array([0., 1., 2.]), array([ 1.,  2., inf]))
    """
    lb, ub = (np.asarray(b, dtype=float) for b in bounds)
    if lb.ndim == 0:
        lb = np.resize(lb, x0.shape)
    if ub.ndim == 0:
        ub = np.resize(ub, x0.shape)
    return (lb, ub)