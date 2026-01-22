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
def _num_features(X):
    """Return the number of features in an array-like X.

    This helper function tries hard to avoid to materialize an array version
    of X unless necessary. For instance, if X is a list of lists,
    this function will return the length of the first element, assuming
    that subsequent elements are all lists of the same length without
    checking.
    Parameters
    ----------
    X : array-like
        array-like to get the number of features.

    Returns
    -------
    features : int
        Number of features
    """
    type_ = type(X)
    if type_.__module__ == 'builtins':
        type_name = type_.__qualname__
    else:
        type_name = f'{type_.__module__}.{type_.__qualname__}'
    message = f'Unable to find the number of features from X of type {type_name}'
    if not hasattr(X, '__len__') and (not hasattr(X, 'shape')):
        if not hasattr(X, '__array__'):
            raise TypeError(message)
        X = np.asarray(X)
    if hasattr(X, 'shape'):
        if not hasattr(X.shape, '__len__') or len(X.shape) <= 1:
            message += f' with shape {X.shape}'
            raise TypeError(message)
        return X.shape[1]
    first_sample = X[0]
    if isinstance(first_sample, (str, bytes, dict)):
        message += f' where the samples are of type {type(first_sample).__qualname__}'
        raise TypeError(message)
    try:
        return len(first_sample)
    except Exception as err:
        raise TypeError(message) from err