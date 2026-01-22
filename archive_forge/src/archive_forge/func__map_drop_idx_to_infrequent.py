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
def _map_drop_idx_to_infrequent(self, feature_idx, drop_idx):
    """Convert `drop_idx` into the index for infrequent categories.

        If there are no infrequent categories, then `drop_idx` is
        returned. This method is called in `_set_drop_idx` when the `drop`
        parameter is an array-like.
        """
    if not self._infrequent_enabled:
        return drop_idx
    default_to_infrequent = self._default_to_infrequent_mappings[feature_idx]
    if default_to_infrequent is None:
        return drop_idx
    infrequent_indices = self._infrequent_indices[feature_idx]
    if infrequent_indices is not None and drop_idx in infrequent_indices:
        categories = self.categories_[feature_idx]
        raise ValueError(f'Unable to drop category {categories[drop_idx].item()!r} from feature {feature_idx} because it is infrequent')
    return default_to_infrequent[drop_idx]