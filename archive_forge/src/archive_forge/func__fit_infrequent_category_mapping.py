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
def _fit_infrequent_category_mapping(self, n_samples, category_counts, missing_indices):
    """Fit infrequent categories.

        Defines the private attribute: `_default_to_infrequent_mappings`. For
        feature `i`, `_default_to_infrequent_mappings[i]` defines the mapping
        from the integer encoding returned by `super().transform()` into
        infrequent categories. If `_default_to_infrequent_mappings[i]` is None,
        there were no infrequent categories in the training set.

        For example if categories 0, 2 and 4 were frequent, while categories
        1, 3, 5 were infrequent for feature 7, then these categories are mapped
        to a single output:
        `_default_to_infrequent_mappings[7] = array([0, 3, 1, 3, 2, 3])`

        Defines private attribute: `_infrequent_indices`. `_infrequent_indices[i]`
        is an array of indices such that
        `categories_[i][_infrequent_indices[i]]` are all the infrequent category
        labels. If the feature `i` has no infrequent categories
        `_infrequent_indices[i]` is None.

        .. versionadded:: 1.1

        Parameters
        ----------
        n_samples : int
            Number of samples in training set.
        category_counts: list of ndarray
            `category_counts[i]` is the category counts corresponding to
            `self.categories_[i]`.
        missing_indices : dict
            Dict mapping from feature_idx to category index with a missing value.
        """
    if missing_indices:
        category_counts_ = []
        for feature_idx, count in enumerate(category_counts):
            if feature_idx in missing_indices:
                category_counts_.append(np.delete(count, missing_indices[feature_idx]))
            else:
                category_counts_.append(count)
    else:
        category_counts_ = category_counts
    self._infrequent_indices = [self._identify_infrequent(category_count, n_samples, col_idx) for col_idx, category_count in enumerate(category_counts_)]
    self._default_to_infrequent_mappings = []
    for feature_idx, infreq_idx in enumerate(self._infrequent_indices):
        cats = self.categories_[feature_idx]
        if infreq_idx is None:
            self._default_to_infrequent_mappings.append(None)
            continue
        n_cats = len(cats)
        if feature_idx in missing_indices:
            n_cats -= 1
        mapping = np.empty(n_cats, dtype=np.int64)
        n_infrequent_cats = infreq_idx.size
        n_frequent_cats = n_cats - n_infrequent_cats
        mapping[infreq_idx] = n_frequent_cats
        frequent_indices = np.setdiff1d(np.arange(n_cats), infreq_idx)
        mapping[frequent_indices] = np.arange(n_frequent_cats)
        self._default_to_infrequent_mappings.append(mapping)