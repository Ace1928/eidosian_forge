from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def _expm_multiply_interval(A, B, start=None, stop=None, num=None, endpoint=None, traceA=None, balance=False, status_only=False):
    """
    Compute the action of the matrix exponential at multiple time points.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable
    endpoint : bool, optional
        If True, `stop` is the last time point. Otherwise, it is not included.
    balance : bool
        Indicates whether or not to apply balancing.
    status_only : bool
        A flag that is set to True for some debugging and testing operations.

    Returns
    -------
    F : ndarray
        :math:`e^{t_k A} B`
    status : int
        An integer status for testing and debugging.

    Notes
    -----
    This is algorithm (5.2) in Al-Mohy and Higham (2011).

    There seems to be a typo, where line 15 of the algorithm should be
    moved to line 6.5 (between lines 6 and 7).

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
        traceA = traceest(A, m3=5) if is_linear_operator else _trace(A)
    mu = traceA / float(n)
    linspace_kwargs = {'retstep': True}
    if num is not None:
        linspace_kwargs['num'] = num
    if endpoint is not None:
        linspace_kwargs['endpoint'] = endpoint
    samples, step = np.linspace(start, stop, **linspace_kwargs)
    nsamples = len(samples)
    if nsamples < 2:
        raise ValueError('at least two time points are required')
    q = nsamples - 1
    h = step
    t_0 = samples[0]
    t_q = samples[q]
    X_shape = (nsamples,) + B.shape
    X = np.empty(X_shape, dtype=np.result_type(A.dtype, B.dtype, float))
    t = t_q - t_0
    A = A - mu * ident
    A_1_norm = onenormest(A) if is_linear_operator else _exact_1_norm(A)
    ell = 2
    norm_info = LazyOperatorNormInfo(t * A, A_1_norm=t * A_1_norm, ell=ell)
    if t * A_1_norm == 0:
        m_star, s = (0, 1)
    else:
        m_star, s = _fragment_3_1(norm_info, n0, tol, ell=ell)
    X[0] = _expm_multiply_simple_core(A, B, t_0, mu, m_star, s)
    if q <= s:
        if status_only:
            return 0
        else:
            return _expm_multiply_interval_core_0(A, X, h, mu, q, norm_info, tol, ell, n0)
    elif not q % s:
        if status_only:
            return 1
        else:
            return _expm_multiply_interval_core_1(A, X, h, mu, m_star, s, q, tol)
    elif q % s:
        if status_only:
            return 2
        else:
            return _expm_multiply_interval_core_2(A, X, h, mu, m_star, s, q, tol)
    else:
        raise Exception('internal error')