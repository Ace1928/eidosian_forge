import warnings
from numbers import Integral, Real
import numpy as np
from .._config import config_context
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..metrics import euclidean_distances, pairwise_distances_argmin
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.validation import check_is_fitted
def _equal_similarities_and_preferences(S, preference):

    def all_equal_preferences():
        return np.all(preference == preference.flat[0])

    def all_equal_similarities():
        mask = np.ones(S.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        return np.all(S[mask].flat == S[mask].flat[0])
    return all_equal_preferences() and all_equal_similarities()