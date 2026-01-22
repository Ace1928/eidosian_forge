import copy
import functools
import inspect
import platform
import re
import warnings
from collections import defaultdict
import numpy as np
from . import __version__
from ._config import config_context, get_config
from .exceptions import InconsistentVersionWarning
from .utils import _IS_32BIT
from .utils._estimator_html_repr import _HTMLDocumentationLinkMixin, estimator_html_repr
from .utils._metadata_requests import _MetadataRequester, _routing_enabled
from .utils._param_validation import validate_parameter_constraints
from .utils._set_output import _SetOutputMixin
from .utils._tags import (
from .utils.validation import (
class OutlierMixin:
    """Mixin class for all outlier detection estimators in scikit-learn.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `outlier_detector`;
    - `fit_predict` method that default to `fit` and `predict`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.base import BaseEstimator, OutlierMixin
    >>> class MyEstimator(OutlierMixin):
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    ...     def predict(self, X):
    ...         return np.ones(shape=len(X))
    >>> estimator = MyEstimator()
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> estimator.fit_predict(X)
    array([1., 1., 1.])
    """
    _estimator_type = 'outlier_detector'

    def fit_predict(self, X, y=None, **kwargs):
        """Perform fit on X and returns labels for X.

        Returns -1 for outliers and 1 for inliers.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        **kwargs : dict
            Arguments to be passed to ``fit``.

            .. versionadded:: 1.4

        Returns
        -------
        y : ndarray of shape (n_samples,)
            1 for inliers, -1 for outliers.
        """
        if _routing_enabled():
            transform_params = self.get_metadata_routing().consumes(method='predict', params=kwargs.keys())
            if transform_params:
                warnings.warn(f"This object ({self.__class__.__name__}) has a `predict` method which consumes metadata, but `fit_predict` does not forward metadata to `predict`. Please implement a custom `fit_predict` method to forward metadata to `predict` as well.Alternatively, you can explicitly do `set_predict_request`and set all values to `False` to disable metadata routed to `predict`, if that's an option.", UserWarning)
        return self.fit(X, **kwargs).predict(X)