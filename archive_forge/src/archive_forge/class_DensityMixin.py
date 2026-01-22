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
class DensityMixin:
    """Mixin class for all density estimators in scikit-learn.

    This mixin defines the following functionality:

    - `_estimator_type` class attribute defaulting to `"DensityEstimator"`;
    - `score` method that default that do no-op.

    Examples
    --------
    >>> from sklearn.base import DensityMixin
    >>> class MyEstimator(DensityMixin):
    ...     def fit(self, X, y=None):
    ...         self.is_fitted_ = True
    ...         return self
    >>> estimator = MyEstimator()
    >>> hasattr(estimator, "score")
    True
    """
    _estimator_type = 'DensityEstimator'

    def score(self, X, y=None):
        """Return the score of the model on the data `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        score : float
        """
        pass