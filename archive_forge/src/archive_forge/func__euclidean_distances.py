import itertools
import warnings
from functools import partial
from numbers import Integral, Real
import numpy as np
from joblib import effective_n_jobs
from scipy.sparse import csr_matrix, issparse
from scipy.spatial import distance
from .. import config_context
from ..exceptions import DataConversionWarning
from ..preprocessing import normalize
from ..utils import (
from ..utils._mask import _get_mask
from ..utils._param_validation import (
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import parse_version, sp_base_version
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _num_samples, check_non_negative
from ._pairwise_distances_reduction import ArgKmin
from ._pairwise_fast import _chi2_kernel_fast, _sparse_manhattan
def _euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None, squared=False):
    """Computational part of euclidean_distances

    Assumes inputs are already checked.

    If norms are passed as float32, they are unused. If arrays are passed as
    float32, norms needs to be recomputed on upcast chunks.
    TODO: use a float64 accumulator in row_norms to avoid the latter.
    """
    if X_norm_squared is not None and X_norm_squared.dtype != np.float32:
        XX = X_norm_squared.reshape(-1, 1)
    elif X.dtype != np.float32:
        XX = row_norms(X, squared=True)[:, np.newaxis]
    else:
        XX = None
    if Y is X:
        YY = None if XX is None else XX.T
    elif Y_norm_squared is not None and Y_norm_squared.dtype != np.float32:
        YY = Y_norm_squared.reshape(1, -1)
    elif Y.dtype != np.float32:
        YY = row_norms(Y, squared=True)[np.newaxis, :]
    else:
        YY = None
    if X.dtype == np.float32 or Y.dtype == np.float32:
        distances = _euclidean_distances_upcast(X, XX, Y, YY)
    else:
        distances = -2 * safe_sparse_dot(X, Y.T, dense_output=True)
        distances += XX
        distances += YY
    np.maximum(distances, 0, out=distances)
    if X is Y:
        np.fill_diagonal(distances, 0)
    return distances if squared else np.sqrt(distances, out=distances)