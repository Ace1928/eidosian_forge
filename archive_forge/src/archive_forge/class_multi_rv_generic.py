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
class multi_rv_generic:
    """
    Class which encapsulates common functionality between all multivariate
    distributions.
    """

    def __init__(self, seed=None):
        super().__init__()
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        """ Get or set the Generator object for generating random variates.

        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.

        """
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def _get_random_state(self, random_state):
        if random_state is not None:
            return check_random_state(random_state)
        else:
            return self._random_state