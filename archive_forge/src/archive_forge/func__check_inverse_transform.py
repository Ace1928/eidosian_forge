import warnings
import numpy as np
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils._param_validation import StrOptions
from ..utils._set_output import ADAPTERS_MANAGER, _get_output_config
from ..utils.metaestimators import available_if
from ..utils.validation import (
def _check_inverse_transform(self, X):
    """Check that func and inverse_func are the inverse."""
    idx_selected = slice(None, None, max(1, X.shape[0] // 100))
    X_round_trip = self.inverse_transform(self.transform(X[idx_selected]))
    if hasattr(X, 'dtype'):
        dtypes = [X.dtype]
    elif hasattr(X, 'dtypes'):
        dtypes = X.dtypes
    if not all((np.issubdtype(d, np.number) for d in dtypes)):
        raise ValueError("'check_inverse' is only supported when all the elements in `X` is numerical.")
    if not _allclose_dense_sparse(X[idx_selected], X_round_trip):
        warnings.warn("The provided functions are not strictly inverse of each other. If you are sure you want to proceed regardless, set 'check_inverse=False'.", UserWarning)