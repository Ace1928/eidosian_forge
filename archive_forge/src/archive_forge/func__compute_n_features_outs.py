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
def _compute_n_features_outs(self):
    """Compute the n_features_out for each input feature."""
    output = [len(cats) for cats in self.categories_]
    if self._drop_idx_after_grouping is not None:
        for i, drop_idx in enumerate(self._drop_idx_after_grouping):
            if drop_idx is not None:
                output[i] -= 1
    if not self._infrequent_enabled:
        return output
    for i, infreq_idx in enumerate(self._infrequent_indices):
        if infreq_idx is None:
            continue
        output[i] -= infreq_idx.size - 1
    return output