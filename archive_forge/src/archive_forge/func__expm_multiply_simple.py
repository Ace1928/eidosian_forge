from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def _expm_multiply_simple(A, B, t=1.0, traceA=None, balance=False):
    """
    Compute the action of the matrix exponential at a single time point.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    t : float
        A time point.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable
    balance : bool
        Indicates whether or not to apply balancing.

    Returns
    -------
    F : ndarray
        :math:`e^{t A} B`

    Notes
    -----
    This is algorithm (3.2) in Al-Mohy and Higham (2011).

    """
    if balance:
        raise NotImplementedError
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be like a square matrix')
    if A.shape[1] != B.shape[0]:
        raise ValueError('shapes of matrices A {} and B {} are incompatible'.format(A.shape, B.shape))
    ident = _ident_like(A)
    is_linear_operator = isinstance(A, scipy.sparse.linalg.LinearOperator)
    n = A.shape[0]
    if len(B.shape) == 1:
        n0 = 1
    elif len(B.shape) == 2:
        n0 = B.shape[1]
    else:
        raise ValueError('expected B to be like a matrix or a vector')
    u_d = 2 ** (-53)
    tol = u_d
    if traceA is None:
        if is_linear_operator:
            warn('Trace of LinearOperator not available, it will be estimated. Provide `traceA` to ensure performance.', stacklevel=3)
        traceA = traceest(A, m3=1) if is_linear_operator else _trace(A)
    mu = traceA / float(n)
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    if t * A_1_norm == 0:
        m_star, s = (0, 1)
    else:
        ell = 2
        norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    return _expm_multiply_simple_core(A, B, t, mu, m_star, s, tol, balance)