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
class unitary_group_gen(multi_rv_generic):
    """A matrix-valued U(N) random variable.

    Return a random unitary matrix.

    The `dim` keyword specifies the dimension N.

    Methods
    -------
    rvs(dim=None, size=1, random_state=None)
        Draw random samples from U(N).

    Parameters
    ----------
    dim : scalar
        Dimension of matrices
    seed : {None, int, np.random.RandomState, np.random.Generator}, optional
        Used for drawing random variates.
        If `seed` is `None`, the `~np.random.RandomState` singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used, seeded
        with seed.
        If `seed` is already a ``RandomState`` or ``Generator`` instance,
        then that object is used.
        Default is `None`.

    Notes
    -----
    This class is similar to `ortho_group`.

    References
    ----------
    .. [1] F. Mezzadri, "How to generate random matrices from the classical
           compact groups", :arXiv:`math-ph/0609050v2`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.stats import unitary_group
    >>> x = unitary_group.rvs(3)

    >>> np.dot(x, x.conj().T)
    array([[  1.00000000e+00,   1.13231364e-17,  -2.86852790e-16],
           [  1.13231364e-17,   1.00000000e+00,  -1.46845020e-16],
           [ -2.86852790e-16,  -1.46845020e-16,   1.00000000e+00]])

    This generates one random matrix from U(3). The dot product confirms that
    it is unitary up to machine precision.

    Alternatively, the object may be called (as a function) to fix the `dim`
    parameter, return a "frozen" unitary_group random variable:

    >>> rv = unitary_group(5)

    See Also
    --------
    ortho_group

    """

    def __init__(self, seed=None):
        super().__init__(seed)
        self.__doc__ = doccer.docformat(self.__doc__)

    def __call__(self, dim=None, seed=None):
        """Create a frozen (U(N)) n-dimensional unitary matrix distribution.

        See `unitary_group_frozen` for more information.
        """
        return unitary_group_frozen(dim, seed=seed)

    def _process_parameters(self, dim):
        """Dimension N must be specified; it cannot be inferred."""
        if dim is None or not np.isscalar(dim) or dim <= 1 or (dim != int(dim)):
            raise ValueError('Dimension of rotation must be specified,and must be a scalar greater than 1.')
        return dim

    def rvs(self, dim, size=1, random_state=None):
        """Draw random samples from U(N).

        Parameters
        ----------
        dim : integer
            Dimension of space (N).
        size : integer, optional
            Number of samples to draw (default 1).

        Returns
        -------
        rvs : ndarray or scalar
            Random size N-dimensional matrices, dimension (size, dim, dim)

        """
        random_state = self._get_random_state(random_state)
        size = int(size)
        dim = self._process_parameters(dim)
        size = (size,) if size > 1 else ()
        z = 1 / math.sqrt(2) * (random_state.normal(size=size + (dim, dim)) + 1j * random_state.normal(size=size + (dim, dim)))
        q, r = np.linalg.qr(z)
        d = r.diagonal(offset=0, axis1=-2, axis2=-1)
        q *= (d / abs(d))[..., np.newaxis, :]
        return q