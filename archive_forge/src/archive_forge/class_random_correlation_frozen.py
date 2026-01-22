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
class random_correlation_frozen(multi_rv_frozen):

    def __init__(self, eigs, seed=None, tol=1e-13, diag_tol=1e-07):
        """Create a frozen random correlation matrix distribution.

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
        """
        self._dist = random_correlation_gen(seed)
        self.tol = tol
        self.diag_tol = diag_tol
        _, self.eigs = self._dist._process_parameters(eigs, tol=self.tol)

    def rvs(self, random_state=None):
        return self._dist.rvs(self.eigs, random_state=random_state, tol=self.tol, diag_tol=self.diag_tol)