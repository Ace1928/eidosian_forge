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
def _get_neighbor_feat_idx(self, n_features, feat_idx, abs_corr_mat):
    """Get a list of other features to predict `feat_idx`.

        If `self.n_nearest_features` is less than or equal to the total
        number of features, then use a probability proportional to the absolute
        correlation between `feat_idx` and each other feature to randomly
        choose a subsample of the other features (without replacement).

        Parameters
        ----------
        n_features : int
            Number of features in `X`.

        feat_idx : int
            Index of the feature currently being imputed.

        abs_corr_mat : ndarray, shape (n_features, n_features)
            Absolute correlation matrix of `X`. The diagonal has been zeroed
            out and each feature has been normalized to sum to 1. Can be None.

        Returns
        -------
        neighbor_feat_idx : array-like
            The features to use to impute `feat_idx`.
        """
    if self.n_nearest_features is not None and self.n_nearest_features < n_features:
        p = abs_corr_mat[:, feat_idx]
        neighbor_feat_idx = self.random_state_.choice(np.arange(n_features), self.n_nearest_features, replace=False, p=p)
    else:
        inds_left = np.arange(feat_idx)
        inds_right = np.arange(feat_idx + 1, n_features)
        neighbor_feat_idx = np.concatenate((inds_left, inds_right))
    return neighbor_feat_idx