import numpy as np
from scipy._lib._util import check_random_state
from scipy.sparse.linalg import aslinearoperator
from scipy.linalg import LinAlgError
from ._propack import _spropack  # type: ignore[attr-defined]
from ._propack import _dpropack  # type: ignore[attr-defined]
from ._propack import _cpropack  # type: ignore[attr-defined]
from ._propack import _zpropack  # type: ignore[attr-defined]

    Compute the singular value decomposition of a linear operator using PROPACK

    Parameters
    ----------
    A : array_like, sparse matrix, or LinearOperator
        Operator for which SVD will be computed.  If `A` is a LinearOperator
        object, it must define both ``matvec`` and ``rmatvec`` methods.
    k : int
        Number of singular values/vectors to compute
    which : {"LM", "SM"}
        Which singular triplets to compute:
        - 'LM': compute triplets corresponding to the `k` largest singular
                values
        - 'SM': compute triplets corresponding to the `k` smallest singular
                values
        `which='SM'` requires `irl_mode=True`.  Computes largest singular
        values by default.
    irl_mode : bool, optional
        If `True`, then compute SVD using IRL (implicitly restarted Lanczos)
        mode.  Default is `True`.
    kmax : int, optional
        Maximal number of iterations / maximal dimension of the Krylov
        subspace. Default is ``10 * k``.
    compute_u : bool, optional
        If `True` (default) then compute left singular vectors, `u`.
    compute_v : bool, optional
        If `True` (default) then compute right singular vectors, `v`.
    tol : float, optional
        The desired relative accuracy for computed singular values.
        If not specified, it will be set based on machine precision.
    v0 : array_like, optional
        Starting vector for iterations: must be of length ``A.shape[0]``.
        If not specified, PROPACK will generate a starting vector.
    full_output : bool, optional
        If `True`, then return sigma_bound.  Default is `False`.
    delta : float, optional
        Level of orthogonality to maintain between Lanczos vectors.
        Default is set based on machine precision.
    eta : float, optional
        Orthogonality cutoff.  During reorthogonalization, vectors with
        component larger than `eta` along the Lanczos vector will be purged.
        Default is set based on machine precision.
    anorm : float, optional
        Estimate of ``||A||``.  Default is `0`.
    cgs : bool, optional
        If `True`, reorthogonalization is done using classical Gram-Schmidt.
        If `False` (default), it is done using modified Gram-Schmidt.
    elr : bool, optional
        If `True` (default), then extended local orthogonality is enforced
        when obtaining singular vectors.
    min_relgap : float, optional
        The smallest relative gap allowed between any shift in IRL mode.
        Default is `0.001`.  Accessed only if ``irl_mode=True``.
    shifts : int, optional
        Number of shifts per restart in IRL mode.  Default is determined
        to satisfy ``k <= min(kmax-shifts, m, n)``.  Must be
        >= 0, but choosing 0 might lead to performance degradation.
        Accessed only if ``irl_mode=True``.
    maxiter : int, optional
        Maximum number of restarts in IRL mode.  Default is `1000`.
        Accessed only if ``irl_mode=True``.
    random_state : {None, int, `numpy.random.Generator`,
                    `numpy.random.RandomState`}, optional

        Pseudorandom number generator state used to generate resamples.

        If `random_state` is ``None`` (or `np.random`), the
        `numpy.random.RandomState` singleton is used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` or ``RandomState``
        instance then that instance is used.

    Returns
    -------
    u : ndarray
        The `k` largest (``which="LM"``) or smallest (``which="SM"``) left
        singular vectors, ``shape == (A.shape[0], 3)``, returned only if
        ``compute_u=True``.
    sigma : ndarray
        The top `k` singular values, ``shape == (k,)``
    vt : ndarray
        The `k` largest (``which="LM"``) or smallest (``which="SM"``) right
        singular vectors, ``shape == (3, A.shape[1])``, returned only if
        ``compute_v=True``.
    sigma_bound : ndarray
        the error bounds on the singular values sigma, returned only if
        ``full_output=True``.

    