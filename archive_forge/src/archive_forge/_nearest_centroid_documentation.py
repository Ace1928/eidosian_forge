import warnings
from numbers import Real
import numpy as np
from scipy import sparse as sp
from sklearn.metrics.pairwise import _VALID_METRICS
from ..base import BaseEstimator, ClassifierMixin, _fit_context
from ..metrics.pairwise import pairwise_distances_argmin
from ..preprocessing import LabelEncoder
from ..utils._param_validation import Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.sparsefuncs import csc_median_axis_0
from ..utils.validation import check_is_fitted
Perform classification on an array of test vectors `X`.

        The predicted class `C` for each sample in `X` is returned.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            The predicted classes.

        Notes
        -----
        If the metric constructor parameter is `"precomputed"`, `X` is assumed
        to be the distance matrix between the data to be predicted and
        `self.centroids_`.
        