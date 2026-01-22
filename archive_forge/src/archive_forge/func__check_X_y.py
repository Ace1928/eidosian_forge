import numpy as np
from ..base import BaseEstimator, ClassifierMixin
from ..utils._metadata_requests import RequestMethod
from .metaestimators import available_if
from .validation import _check_sample_weight, _num_samples, check_array, check_is_fitted
def _check_X_y(self, X, y=None, should_be_fitted=True):
    """Validate X and y and make extra check.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data set.
            `X` is checked only if `check_X` is not `None` (default is None).
        y : array-like of shape (n_samples), default=None
            The corresponding target, by default `None`.
            `y` is checked only if `check_y` is not `None` (default is None).
        should_be_fitted : bool, default=True
            Whether or not the classifier should be already fitted.
            By default True.

        Returns
        -------
        X, y
        """
    if should_be_fitted:
        check_is_fitted(self)
    if self.check_X is not None:
        params = {} if self.check_X_params is None else self.check_X_params
        checked_X = self.check_X(X, **params)
        if isinstance(checked_X, (bool, np.bool_)):
            assert checked_X
        else:
            X = checked_X
    if y is not None and self.check_y is not None:
        params = {} if self.check_y_params is None else self.check_y_params
        checked_y = self.check_y(y, **params)
        if isinstance(checked_y, (bool, np.bool_)):
            assert checked_y
        else:
            y = checked_y
    return (X, y)