import numbers
import operator
import sys
import warnings
from contextlib import suppress
from functools import reduce, wraps
from inspect import Parameter, isclass, signature
import joblib
import numpy as np
import scipy.sparse as sp
from .. import get_config as _get_config
from ..exceptions import DataConversionWarning, NotFittedError, PositiveSpectrumWarning
from ..utils._array_api import _asarray_with_order, _is_numpy_namespace, get_namespace
from ..utils.fixes import ComplexWarning, _preserve_dia_indices_dtype
from ._isfinite import FiniteStatus, cy_isfinite
from .fixes import _object_dtype_isnan
def _check_method_params(X, params, indices=None):
    """Check and validate the parameters passed to a specific
    method like `fit`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data array.

    params : dict
        Dictionary containing the parameters passed to the method.

    indices : array-like of shape (n_samples,), default=None
        Indices to be selected if the parameter has the same size as `X`.

    Returns
    -------
    method_params_validated : dict
        Validated parameters. We ensure that the values support indexing.
    """
    from . import _safe_indexing
    method_params_validated = {}
    for param_key, param_value in params.items():
        if not _is_arraylike(param_value) or _num_samples(param_value) != _num_samples(X):
            method_params_validated[param_key] = param_value
        else:
            method_params_validated[param_key] = _make_indexable(param_value)
            method_params_validated[param_key] = _safe_indexing(method_params_validated[param_key], indices)
    return method_params_validated