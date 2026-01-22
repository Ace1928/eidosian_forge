import collections
from itertools import chain, combinations
from itertools import combinations_with_replacement as combinations_w_r
from numbers import Integral
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import comb
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array
from ..utils._param_validation import Interval, StrOptions
from ..utils.fixes import parse_version, sp_version
from ..utils.stats import _weighted_percentile
from ..utils.validation import (
from ._csr_polynomial_expansion import (
def _create_expansion(X, interaction_only, deg, n_features, cumulative_size=0):
    """Helper function for creating and appending sparse expansion matrices"""
    total_nnz = _calc_total_nnz(X.indptr, interaction_only, deg)
    expanded_col = _calc_expanded_nnz(n_features, interaction_only, deg)
    if expanded_col == 0:
        return None
    max_indices = expanded_col - 1
    max_indptr = total_nnz
    max_int32 = np.iinfo(np.int32).max
    needs_int64 = max(max_indices, max_indptr) > max_int32
    index_dtype = np.int64 if needs_int64 else np.int32
    cumulative_size += expanded_col
    if sp_version < parse_version('1.8.0') and cumulative_size - 1 > max_int32 and (not needs_int64):
        raise ValueError('In scipy versions `<1.8.0`, the function `scipy.sparse.hstack` sometimes produces negative columns when the output shape contains `n_cols` too large to be represented by a 32bit signed integer. To avoid this error, either use a version of scipy `>=1.8.0` or alter the `PolynomialFeatures` transformer to produce fewer than 2^31 output features.')
    expanded_data = np.empty(shape=total_nnz, dtype=X.data.dtype)
    expanded_indices = np.empty(shape=total_nnz, dtype=index_dtype)
    expanded_indptr = np.empty(shape=X.indptr.shape[0], dtype=index_dtype)
    _csr_polynomial_expansion(X.data, X.indices, X.indptr, X.shape[1], expanded_data, expanded_indices, expanded_indptr, interaction_only, deg)
    return sparse.csr_matrix((expanded_data, expanded_indices, expanded_indptr), shape=(X.indptr.shape[0] - 1, expanded_col), dtype=X.dtype)