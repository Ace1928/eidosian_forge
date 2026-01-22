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
def _get_ordered_idx(self, mask_missing_values):
    """Decide in what order we will update the features.

        As a homage to the MICE R package, we will have 4 main options of
        how to order the updates, and use a random order if anything else
        is specified.

        Also, this function skips features which have no missing values.

        Parameters
        ----------
        mask_missing_values : array-like, shape (n_samples, n_features)
            Input data's missing indicator matrix, where `n_samples` is the
            number of samples and `n_features` is the number of features.

        Returns
        -------
        ordered_idx : ndarray, shape (n_features,)
            The order in which to impute the features.
        """
    frac_of_missing_values = mask_missing_values.mean(axis=0)
    if self.skip_complete:
        missing_values_idx = np.flatnonzero(frac_of_missing_values)
    else:
        missing_values_idx = np.arange(np.shape(frac_of_missing_values)[0])
    if self.imputation_order == 'roman':
        ordered_idx = missing_values_idx
    elif self.imputation_order == 'arabic':
        ordered_idx = missing_values_idx[::-1]
    elif self.imputation_order == 'ascending':
        n = len(frac_of_missing_values) - len(missing_values_idx)
        ordered_idx = np.argsort(frac_of_missing_values, kind='mergesort')[n:]
    elif self.imputation_order == 'descending':
        n = len(frac_of_missing_values) - len(missing_values_idx)
        ordered_idx = np.argsort(frac_of_missing_values, kind='mergesort')[n:][::-1]
    elif self.imputation_order == 'random':
        ordered_idx = missing_values_idx
        self.random_state_.shuffle(ordered_idx)
    return ordered_idx