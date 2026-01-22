from numbers import Real
import numpy as np
from ..base import BaseEstimator, _fit_context
from ..utils._param_validation import Interval
from ..utils.sparsefuncs import mean_variance_axis, min_max_axis
from ..utils.validation import check_is_fitted
from ._base import SelectorMixin
Learn empirical variances from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data from which to compute variances, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        y : any, default=None
            Ignored. This parameter exists only for compatibility with
            sklearn.pipeline.Pipeline.

        Returns
        -------
        self : object
            Returns the instance itself.
        