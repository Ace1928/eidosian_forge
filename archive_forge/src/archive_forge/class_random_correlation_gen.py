import math
import numpy as np
import scipy.linalg
from scipy._lib import doccer
from scipy.special import (gammaln, psi, multigammaln, xlogy, entr, betaln,
from scipy._lib._util import check_random_state, _lazywhere
from scipy.linalg.blas import drot, get_blas_funcs
from ._continuous_distns import norm
from ._discrete_distns import binom
from . import _mvn, _covariance, _rcont
from ._qmvnt import _qmvt
from ._morestats import directional_stats
from scipy.optimize import root_scalar
class random_correlation_gen(multi_rv_generic):
    """A random correlation matrix.

    Return a random correlation matrix, given a vector of eigenvalues.

    The `eigs` keyword specifies the eigenvalues of the correlation matrix,
    and implies the dimension.

    Methods
    -------
    rvs(eigs=None, random_state=None)
        Draw random correlation matrices, all with eigenvalues eigs.

    Parameters
    ----------
    eigs : 1d ndarray
        Eigenvalues of correlation matrix
    seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance
        then that instance is used.
    tol : float, optional
        Tolerance for input parameter checks
    diag_tol : float, optional
        Tolerance for deviation of the diagonal of the resulting
        matrix. Default: 1e-7

    Raises
    ------
    RuntimeError
        Floating point error prevented generating a valid correlation
        matrix.

    Returns
    -------
    rvs : ndarray or scalar
        Random size N-dimensional matrices, dimension (size, dim, dim),
        each having eigenvalues eigs.

    Notes
    -----

    Generates a random correlation matrix following a numerically stable
    algorithm spelled out by Davies & Higham. This algorithm uses a single O(N)
    similarity transformation to construct a symmetric positive semi-definite
    matrix, and applies a series of Givens rotations to scale it to have ones
    on the diagonal.

    References
    ----------

    .. [1] Davies, Philip I; Higham, Nicholas J; "Numerically stable generation
           of correlation matrices and their factors", BIT 2000, Vol. 40,
           No. 4, pp. 640 651

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import random_correlation
    >>> rng = np.random.default_rng()
    >>> x = random_correlation.rvs((.5, .8, 1.2, 1.5), random_state=rng)
    >>> x
    array([[ 1.        , -0.02423399,  0.03130519,  0.4946965 ],
           [-0.02423399,  1.        ,  0.20334736,  0.04039817],
           [ 0.03130519,  0.20334736,  1.        ,  0.02694275],
           [ 0.4946965 ,  0.04039817,  0.02694275,  1.        ]])
    >>> import scipy.linalg
    >>> e, v = scipy.linalg.eigh(x)
    >>> e
    array([ 0.5,  0.8,  1.2,  1.5])

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, eigs, seed=None, tol=1e-13, diag_tol=1e-07):
        """Create a frozen random correlation matrix.

        See `random_correlation_frozen` for more information.
        """
        return random_correlation_frozen(eigs, seed=seed, tol=tol, diag_tol=diag_tol)

    def _process_parameters(self, eigs, tol):
        eigs = np.asarray(eigs, dtype=float)
        dim = eigs.size
        if eigs.ndim != 1 or eigs.shape[0] != dim or dim <= 1:
            raise ValueError("Array 'eigs' must be a vector of length greater than 1.")
        if np.fabs(np.sum(eigs) - dim) > tol:
            raise ValueError('Sum of eigenvalues must equal dimensionality.')
        for x in eigs:
            if x < -tol:
                raise ValueError('All eigenvalues must be non-negative.')
        return (dim, eigs)

    def _givens_to_1(self, aii, ajj, aij):
        """Computes a 2x2 Givens matrix to put 1's on the diagonal.

        The input matrix is a 2x2 symmetric matrix M = [ aii aij ; aij ajj ].

        The output matrix g is a 2x2 anti-symmetric matrix of the form
        [ c s ; -s c ];  the elements c and s are returned.

        Applying the output matrix to the input matrix (as b=g.T M g)
        results in a matrix with bii=1, provided tr(M) - det(M) >= 1
        and floating point issues do not occur. Otherwise, some other
        valid rotation is returned. When tr(M)==2, also bjj=1.

        """
        aiid = aii - 1.0
        ajjd = ajj - 1.0
        if ajjd == 0:
            return (0.0, 1.0)
        dd = math.sqrt(max(aij ** 2 - aiid * ajjd, 0))
        t = (aij + math.copysign(dd, aij)) / ajjd
        c = 1.0 / math.sqrt(1.0 + t * t)
        if c == 0:
            s = 1.0
        else:
            s = c * t
        return (c, s)

    def _to_corr(self, m):
        """
        Given a psd matrix m, rotate to put one's on the diagonal, turning it
        into a correlation matrix.  This also requires the trace equal the
        dimensionality. Note: modifies input matrix
        """
        if not (m.flags.c_contiguous and m.dtype == np.float64 and (m.shape[0] == m.shape[1])):
            raise ValueError()
        d = m.shape[0]
        for i in range(d - 1):
            if m[i, i] == 1:
                continue
            elif m[i, i] > 1:
                for j in range(i + 1, d):
                    if m[j, j] < 1:
                        break
            else:
                for j in range(i + 1, d):
                    if m[j, j] > 1:
                        break
            c, s = self._givens_to_1(m[i, i], m[j, j], m[i, j])
            mv = m.ravel()
            drot(mv, mv, c, -s, n=d, offx=i * d, incx=1, offy=j * d, incy=1, overwrite_x=True, overwrite_y=True)
            drot(mv, mv, c, -s, n=d, offx=i, incx=d, offy=j, incy=d, overwrite_x=True, overwrite_y=True)
        return m

    def rvs(self, eigs, random_state=None, tol=1e-13, diag_tol=1e-07):
        """Draw random correlation matrices.

        Parameters
        ----------
        eigs : 1d ndarray
            Eigenvalues of correlation matrix
        tol : float, optional
            Tolerance for input parameter checks
        diag_tol : float, optional
            Tolerance for deviation of the diagonal of the resulting
            matrix. Default: 1e-7

        Raises
        ------
        RuntimeError
            Floating point error prevented generating a valid correlation
            matrix.

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim),
            each having eigenvalues eigs.

        """
        dim, eigs = self._process_parameters(eigs, tol=tol)
        random_state = self._get_random_state(random_state)
        m = ortho_group.rvs(dim, random_state=random_state)
        m = np.dot(np.dot(m, np.diag(eigs)), m.T)
        m = self._to_corr(m)
        if abs(m.diagonal() - 1).max() > diag_tol:
            raise RuntimeError('Failed to generate a valid correlation matrix')
        return m