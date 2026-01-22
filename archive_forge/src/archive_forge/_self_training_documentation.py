import warnings
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone
from ..utils import safe_mask
from ..utils._param_validation import HasMethods, Interval, StrOptions
from ..utils.metadata_routing import _RoutingNotSupportedMixin
from ..utils.metaestimators import available_if
from ..utils.validation import check_is_fitted
Call score on the `base_estimator`.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.

        y : array-like of shape (n_samples,)
            Array representing the labels.

        Returns
        -------
        score : float
            Result of calling score on the `base_estimator`.
        