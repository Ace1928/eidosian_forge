import numbers
from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import issparse
from ..base import OutlierMixin, _fit_context
from ..tree import ExtraTreeRegressor
from ..tree._tree import DTYPE as tree_dtype
from ..utils import (
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.validation import _num_samples, check_is_fitted
from ._bagging import BaseBagging
def _compute_score_samples(self, X, subsample_features):
    """
        Compute the score of each samples in X going through the extra trees.

        Parameters
        ----------
        X : array-like or sparse matrix
            Data matrix.

        subsample_features : bool
            Whether features should be subsampled.
        """
    n_samples = X.shape[0]
    depths = np.zeros(n_samples, order='f')
    average_path_length_max_samples = _average_path_length([self._max_samples])
    for tree_idx, (tree, features) in enumerate(zip(self.estimators_, self.estimators_features_)):
        X_subset = X[:, features] if subsample_features else X
        leaves_index = tree.apply(X_subset, check_input=False)
        depths += self._decision_path_lengths[tree_idx][leaves_index] + self._average_path_length_per_tree[tree_idx][leaves_index] - 1.0
    denominator = len(self.estimators_) * average_path_length_max_samples
    scores = 2 ** (-np.divide(depths, denominator, out=np.ones_like(depths), where=denominator != 0))
    return scores