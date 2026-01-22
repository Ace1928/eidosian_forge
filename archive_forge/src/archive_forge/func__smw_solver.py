import warnings
import numpy as np
import pandas as pd
import patsy
from scipy import sparse
from scipy.stats.distributions import norm
from statsmodels.base._penalties import Penalty
import statsmodels.base.model as base
from statsmodels.tools import data as data_tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
def _smw_solver(s, A, AtA, Qi, di):
    """
    Returns a solver for the linear system:

    .. math::

        (sI + ABA^\\prime) y = x

    The returned function f satisfies f(x) = y as defined above.

    B and its inverse matrix are block diagonal.  The upper left block
    of :math:`B^{-1}` is Qi and its lower right block is diag(di).

    Parameters
    ----------
    s : scalar
        See above for usage
    A : ndarray
        p x q matrix, in general q << p, may be sparse.
    AtA : square ndarray
        :math:`A^\\prime  A`, a q x q matrix.
    Qi : square symmetric ndarray
        The matrix `B` is q x q, where q = r + d.  `B` consists of a r
        x r diagonal block whose inverse is `Qi`, and a d x d diagonal
        block, whose inverse is diag(di).
    di : 1d array_like
        See documentation for Qi.

    Returns
    -------
    A function for solving a linear system, as documented above.

    Notes
    -----
    Uses Sherman-Morrison-Woodbury identity:
        https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    """
    qmat = AtA / s
    m = Qi.shape[0]
    qmat[0:m, 0:m] += Qi
    if sparse.issparse(A):
        qmat[m:, m:] += sparse.diags(di)

        def solver(rhs):
            ql = A.T.dot(rhs)
            ql = sparse.linalg.spsolve(qmat, ql)
            if ql.ndim < rhs.ndim:
                ql = ql[:, None]
            ql = A.dot(ql)
            return rhs / s - ql / s ** 2
    else:
        d = qmat.shape[0]
        qmat.flat[m * (d + 1)::d + 1] += di
        qmati = np.linalg.solve(qmat, A.T)

        def solver(rhs):
            ql = np.dot(qmati, rhs)
            ql = np.dot(A, ql)
            return rhs / s - ql / s ** 2
    return solver