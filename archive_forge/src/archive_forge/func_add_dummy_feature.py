import warnings
from numbers import Integral, Real
import numpy as np
from scipy import optimize, sparse, stats
from scipy.special import boxcox
from ..base import (
from ..utils import _array_api, check_array
from ..utils._array_api import get_namespace
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.extmath import _incremental_mean_and_var, row_norms
from ..utils.sparsefuncs import (
from ..utils.sparsefuncs_fast import (
from ..utils.validation import (
from ._encoders import OneHotEncoder
@validate_params({'X': ['array-like', 'sparse matrix'], 'value': [Interval(Real, None, None, closed='neither')]}, prefer_skip_nested_validation=True)
def add_dummy_feature(X, value=1.0):
    """Augment dataset with an additional dummy feature.

    This is useful for fitting an intercept term with implementations which
    cannot otherwise fit it directly.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Data.

    value : float
        Value to use for the dummy feature.

    Returns
    -------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features + 1)
        Same data with dummy feature added as first column.

    Examples
    --------
    >>> from sklearn.preprocessing import add_dummy_feature
    >>> add_dummy_feature([[0, 1], [1, 0]])
    array([[1., 0., 1.],
           [1., 1., 0.]])
    """
    X = check_array(X, accept_sparse=['csc', 'csr', 'coo'], dtype=FLOAT_DTYPES)
    n_samples, n_features = X.shape
    shape = (n_samples, n_features + 1)
    if sparse.issparse(X):
        if X.format == 'coo':
            col = X.col + 1
            col = np.concatenate((np.zeros(n_samples), col))
            row = np.concatenate((np.arange(n_samples), X.row))
            data = np.concatenate((np.full(n_samples, value), X.data))
            return sparse.coo_matrix((data, (row, col)), shape)
        elif X.format == 'csc':
            indptr = X.indptr + n_samples
            indptr = np.concatenate((np.array([0]), indptr))
            indices = np.concatenate((np.arange(n_samples), X.indices))
            data = np.concatenate((np.full(n_samples, value), X.data))
            return sparse.csc_matrix((data, indices, indptr), shape)
        else:
            klass = X.__class__
            return klass(add_dummy_feature(X.tocoo(), value))
    else:
        return np.hstack((np.full((n_samples, 1), value), X))