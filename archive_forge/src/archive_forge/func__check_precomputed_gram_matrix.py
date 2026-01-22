import numbers
import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from scipy import linalg, optimize, sparse
from scipy.sparse.linalg import lsqr
from scipy.special import expit
from ..base import (
from ..utils import check_array, check_random_state
from ..utils._array_api import get_namespace
from ..utils._seq_dataset import (
from ..utils.extmath import safe_sparse_dot
from ..utils.parallel import Parallel, delayed
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import FLOAT_DTYPES, _check_sample_weight, check_is_fitted
def _check_precomputed_gram_matrix(X, precompute, X_offset, X_scale, rtol=None, atol=1e-05):
    """Computes a single element of the gram matrix and compares it to
    the corresponding element of the user supplied gram matrix.

    If the values do not match a ValueError will be thrown.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data array.

    precompute : array-like of shape (n_features, n_features)
        User-supplied gram matrix.

    X_offset : ndarray of shape (n_features,)
        Array of feature means used to center design matrix.

    X_scale : ndarray of shape (n_features,)
        Array of feature scale factors used to normalize design matrix.

    rtol : float, default=None
        Relative tolerance; see numpy.allclose
        If None, it is set to 1e-4 for arrays of dtype numpy.float32 and 1e-7
        otherwise.

    atol : float, default=1e-5
        absolute tolerance; see :func`numpy.allclose`. Note that the default
        here is more tolerant than the default for
        :func:`numpy.testing.assert_allclose`, where `atol=0`.

    Raises
    ------
    ValueError
        Raised when the provided Gram matrix is not consistent.
    """
    n_features = X.shape[1]
    f1 = n_features // 2
    f2 = min(f1 + 1, n_features - 1)
    v1 = (X[:, f1] - X_offset[f1]) * X_scale[f1]
    v2 = (X[:, f2] - X_offset[f2]) * X_scale[f2]
    expected = np.dot(v1, v2)
    actual = precompute[f1, f2]
    dtypes = [precompute.dtype, expected.dtype]
    if rtol is None:
        rtols = [0.0001 if dtype == np.float32 else 1e-07 for dtype in dtypes]
        rtol = max(rtols)
    if not np.isclose(expected, actual, rtol=rtol, atol=atol):
        raise ValueError(f"Gram matrix passed in via 'precompute' parameter did not pass validation when a single element was checked - please check that it was computed properly. For element ({f1},{f2}) we computed {expected} but the user-supplied value was {actual}.")