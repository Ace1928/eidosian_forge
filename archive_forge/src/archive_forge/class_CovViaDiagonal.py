from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
class CovViaDiagonal(Covariance):

    def __init__(self, diagonal):
        diagonal = self._validate_vector(diagonal, 'diagonal')
        i_zero = diagonal <= 0
        positive_diagonal = np.array(diagonal, dtype=np.float64)
        positive_diagonal[i_zero] = 1
        self._log_pdet = np.sum(np.log(positive_diagonal), axis=-1)
        psuedo_reciprocals = 1 / np.sqrt(positive_diagonal)
        psuedo_reciprocals[i_zero] = 0
        self._sqrt_diagonal = np.sqrt(diagonal)
        self._LP = psuedo_reciprocals
        self._rank = positive_diagonal.shape[-1] - i_zero.sum(axis=-1)
        self._covariance = np.apply_along_axis(np.diag, -1, diagonal)
        self._i_zero = i_zero
        self._shape = self._covariance.shape
        self._allow_singular = True

    def _whiten(self, x):
        return _dot_diag(x, self._LP)

    def _colorize(self, x):
        return _dot_diag(x, self._sqrt_diagonal)

    def _support_mask(self, x):
        """
        Check whether x lies in the support of the distribution.
        """
        return ~np.any(_dot_diag(x, self._i_zero), axis=-1)