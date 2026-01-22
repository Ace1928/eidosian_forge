import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def _fill_missing_em(self):
    """
        EM algorithm to fill missing values
        """
    non_missing = np.logical_not(np.isnan(self.data))
    if np.all(non_missing):
        return self.data
    data = self.transformed_data = np.asarray(self._prepare_data())
    ncomp = self._ncomp
    col_non_missing = np.sum(non_missing, 1)
    row_non_missing = np.sum(non_missing, 0)
    if np.any(col_non_missing < ncomp) or np.any(row_non_missing < ncomp):
        raise ValueError('Implementation requires that all columns and all rows have at least ncomp non-missing values')
    mask = np.isnan(data)
    mu = np.nanmean(data, 0)
    projection = np.ones((self._nobs, 1)) * mu
    projection_masked = projection[mask]
    data[mask] = projection_masked
    diff = 1.0
    _iter = 0
    while diff > self._tol_em and _iter < self._max_em_iter:
        last_projection_masked = projection_masked
        self.transformed_data = data
        self._compute_eig()
        self._compute_pca_from_eig()
        projection = np.asarray(self.project(transform=False, unweight=False))
        projection_masked = projection[mask]
        data[mask] = projection_masked
        delta = last_projection_masked - projection_masked
        diff = _norm(delta) / _norm(projection_masked)
        _iter += 1
    data = self._adjusted_data + 0.0
    projection = np.asarray(self.project())
    data[mask] = projection[mask]
    return data