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
def _check_response_method(estimator, response_method):
    """Check if `response_method` is available in estimator and return it.

    .. versionadded:: 1.3

    Parameters
    ----------
    estimator : estimator instance
        Classifier or regressor to check.

    response_method : {"predict_proba", "predict_log_proba", "decision_function",
            "predict"} or list of such str
        Specifies the response method to use get prediction from an estimator
        (i.e. :term:`predict_proba`, :term:`predict_log_proba`,
        :term:`decision_function` or :term:`predict`). Possible choices are:
        - if `str`, it corresponds to the name to the method to return;
        - if a list of `str`, it provides the method names in order of
          preference. The method returned corresponds to the first method in
          the list and which is implemented by `estimator`.

    Returns
    -------
    prediction_method : callable
        Prediction method of estimator.

    Raises
    ------
    AttributeError
        If `response_method` is not available in `estimator`.
    """
    if isinstance(response_method, str):
        list_methods = [response_method]
    else:
        list_methods = response_method
    prediction_method = [getattr(estimator, method, None) for method in list_methods]
    prediction_method = reduce(lambda x, y: x or y, prediction_method)
    if prediction_method is None:
        raise AttributeError(f'{estimator.__class__.__name__} has none of the following attributes: {', '.join(list_methods)}.')
    return prediction_method