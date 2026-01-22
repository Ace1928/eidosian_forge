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
def _pairwise_callable(X, Y, metric, force_all_finite=True, **kwds):
    """Handle the callable case for pairwise_{distances,kernels}."""
    X, Y = check_pairwise_arrays(X, Y, force_all_finite=force_all_finite)
    if X is Y:
        out = np.zeros((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.combinations(range(X.shape[0]), 2)
        for i, j in iterator:
            x = X[[i], :] if issparse(X) else X[i]
            y = Y[[j], :] if issparse(Y) else Y[j]
            out[i, j] = metric(x, y, **kwds)
        out = out + out.T
        for i in range(X.shape[0]):
            x = X[[i], :] if issparse(X) else X[i]
            out[i, i] = metric(x, x, **kwds)
    else:
        out = np.empty((X.shape[0], Y.shape[0]), dtype='float')
        iterator = itertools.product(range(X.shape[0]), range(Y.shape[0]))
        for i, j in iterator:
            x = X[[i], :] if issparse(X) else X[i]
            y = Y[[j], :] if issparse(Y) else Y[j]
            out[i, j] = metric(x, y, **kwds)
    return out