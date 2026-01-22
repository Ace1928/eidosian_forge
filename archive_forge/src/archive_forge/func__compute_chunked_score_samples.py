import numbers
from numbers import Integral, Real
from warnings import warn
import numpy as np
from scipy.sparse import issparse
from ..base import OutlierMixin, _fit_context
from ..tree import ExtraTreeRegressor
from ..tree._tree import DTYPE as tree_dtype
from ..utils import (
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.validation import _num_samples, check_is_fitted
from ._bagging import BaseBagging
def _compute_chunked_score_samples(self, X):
    n_samples = _num_samples(X)
    if self._max_features == X.shape[1]:
        subsample_features = False
    else:
        subsample_features = True
    chunk_n_rows = get_chunk_n_rows(row_bytes=16 * self._max_features, max_n_rows=n_samples)
    slices = gen_batches(n_samples, chunk_n_rows)
    scores = np.zeros(n_samples, order='f')
    for sl in slices:
        scores[sl] = self._compute_score_samples(X[sl], subsample_features)
    return scores