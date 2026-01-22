import warnings
from collections import namedtuple
from numbers import Integral, Real
from time import time
import numpy as np
from scipy import stats
from ..base import _fit_context, clone
from ..exceptions import ConvergenceWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.validation import FLOAT_DTYPES, _check_feature_names_in, check_is_fitted
from ._base import SimpleImputer, _BaseImputer, _check_inputs_dtype
def _get_abs_corr_mat(self, X_filled, tolerance=1e-06):
    """Get absolute correlation matrix between features.

        Parameters
        ----------
        X_filled : ndarray, shape (n_samples, n_features)
            Input data with the most recent imputations.

        tolerance : float, default=1e-6
            `abs_corr_mat` can have nans, which will be replaced
            with `tolerance`.

        Returns
        -------
        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of `X` at the beginning of the
            current round. The diagonal has been zeroed out and each feature's
            absolute correlations with all others have been normalized to sum
            to 1.
        """
    n_features = X_filled.shape[1]
    if self.n_nearest_features is None or self.n_nearest_features >= n_features:
        return None
    with np.errstate(invalid='ignore'):
        abs_corr_mat = np.abs(np.corrcoef(X_filled.T))
    abs_corr_mat[np.isnan(abs_corr_mat)] = tolerance
    np.clip(abs_corr_mat, tolerance, None, out=abs_corr_mat)
    np.fill_diagonal(abs_corr_mat, 0)
    abs_corr_mat = normalize(abs_corr_mat, norm='l1', axis=0, copy=False)
    return abs_corr_mat