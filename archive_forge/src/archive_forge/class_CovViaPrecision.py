from functools import cached_property
import numpy as np
from scipy import linalg
from scipy.stats import _multivariate
class CovViaPrecision(Covariance):

    def __init__(self, precision, covariance=None):
        precision = self._validate_matrix(precision, 'precision')
        if covariance is not None:
            covariance = self._validate_matrix(covariance, 'covariance')
            message = '`precision.shape` must equal `covariance.shape`.'
            if precision.shape != covariance.shape:
                raise ValueError(message)
        self._chol_P = np.linalg.cholesky(precision)
        self._log_pdet = -2 * np.log(np.diag(self._chol_P)).sum(axis=-1)
        self._rank = precision.shape[-1]
        self._precision = precision
        self._cov_matrix = covariance
        self._shape = precision.shape
        self._allow_singular = False

    def _whiten(self, x):
        return x @ self._chol_P

    @cached_property
    def _covariance(self):
        n = self._shape[-1]
        return linalg.cho_solve((self._chol_P, True), np.eye(n)) if self._cov_matrix is None else self._cov_matrix

    def _colorize(self, x):
        return linalg.solve_triangular(self._chol_P.T, x.T, lower=False).T