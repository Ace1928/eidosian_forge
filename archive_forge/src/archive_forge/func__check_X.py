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
def _check_X(self, X, force_all_finite=True):
    """
        Perform custom check_array:
        - convert list of strings to object dtype
        - check for missing values for object dtype data (check_array does
          not do that)
        - return list of features (arrays): this list of features is
          constructed feature by feature to preserve the data types
          of pandas DataFrame columns, as otherwise information is lost
          and cannot be used, e.g. for the `categories_` attribute.

        """
    if not (hasattr(X, 'iloc') and getattr(X, 'ndim', 0) == 2):
        X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, np.str_):
            X = check_array(X, dtype=object, force_all_finite=force_all_finite)
        else:
            X = X_temp
        needs_validation = False
    else:
        needs_validation = force_all_finite
    n_samples, n_features = X.shape
    X_columns = []
    for i in range(n_features):
        Xi = _safe_indexing(X, indices=i, axis=1)
        Xi = check_array(Xi, ensure_2d=False, dtype=None, force_all_finite=needs_validation)
        X_columns.append(Xi)
    return (X_columns, n_samples, n_features)