import numbers
import sys
import warnings
from abc import ABC, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy import sparse
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..model_selection import check_cv
from ..utils import Bunch, check_array, check_scalar
from ..utils._metadata_requests import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import safe_sparse_dot
from ..utils.metadata_routing import (
from ..utils.parallel import Parallel, delayed
from ..utils.validation import (
from . import _cd_fast as cd_fast  # type: ignore
from ._base import LinearModel, _pre_fit, _preprocess_data
def _set_order(X, y, order='C'):
    """Change the order of X and y if necessary.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : ndarray of shape (n_samples,)
        Target values.

    order : {None, 'C', 'F'}
        If 'C', dense arrays are returned as C-ordered, sparse matrices in csr
        format. If 'F', dense arrays are return as F-ordered, sparse matrices
        in csc format.

    Returns
    -------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data with guaranteed order.

    y : ndarray of shape (n_samples,)
        Target values with guaranteed order.
    """
    if order not in [None, 'C', 'F']:
        raise ValueError("Unknown value for order. Got {} instead of None, 'C' or 'F'.".format(order))
    sparse_X = sparse.issparse(X)
    sparse_y = sparse.issparse(y)
    if order is not None:
        sparse_format = 'csc' if order == 'F' else 'csr'
        if sparse_X:
            X = X.asformat(sparse_format, copy=False)
        else:
            X = np.asarray(X, order=order)
        if sparse_y:
            y = y.asformat(sparse_format)
        else:
            y = np.asarray(y, order=order)
    return (X, y)