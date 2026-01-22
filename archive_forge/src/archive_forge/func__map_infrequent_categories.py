import numbers
import warnings
from numbers import Integral
import numpy as np
from scipy import sparse
from ..base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin, _fit_context
from ..utils import _safe_indexing, check_array, is_scalar_nan
from ..utils._encode import _check_unknown, _encode, _get_counts, _unique
from ..utils._mask import _get_mask
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils._set_output import _get_output_config
from ..utils.validation import _check_feature_names_in, check_is_fitted
def _map_infrequent_categories(self, X_int, X_mask, ignore_category_indices):
    """Map infrequent categories to integer representing the infrequent category.

        This modifies X_int in-place. Values that were invalid based on `X_mask`
        are mapped to the infrequent category if there was an infrequent
        category for that feature.

        Parameters
        ----------
        X_int: ndarray of shape (n_samples, n_features)
            Integer encoded categories.

        X_mask: ndarray of shape (n_samples, n_features)
            Bool mask for valid values in `X_int`.

        ignore_category_indices : dict
            Dictionary mapping from feature_idx to category index to ignore.
            Ignored indexes will not be grouped and the original ordinal encoding
            will remain.
        """
    if not self._infrequent_enabled:
        return
    ignore_category_indices = ignore_category_indices or {}
    for col_idx in range(X_int.shape[1]):
        infrequent_idx = self._infrequent_indices[col_idx]
        if infrequent_idx is None:
            continue
        X_int[~X_mask[:, col_idx], col_idx] = infrequent_idx[0]
        if self.handle_unknown == 'infrequent_if_exist':
            X_mask[:, col_idx] = True
    for i, mapping in enumerate(self._default_to_infrequent_mappings):
        if mapping is None:
            continue
        if i in ignore_category_indices:
            rows_to_update = X_int[:, i] != ignore_category_indices[i]
        else:
            rows_to_update = slice(None)
        X_int[rows_to_update, i] = np.take(mapping, X_int[rows_to_update, i])