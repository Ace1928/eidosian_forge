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
def _check_chunk_size(reduced, chunk_size):
    """Checks chunk is a sequence of expected size or a tuple of same."""
    if reduced is None:
        return
    is_tuple = isinstance(reduced, tuple)
    if not is_tuple:
        reduced = (reduced,)
    if any((isinstance(r, tuple) or not hasattr(r, '__iter__') for r in reduced)):
        raise TypeError('reduce_func returned %r. Expected sequence(s) of length %d.' % (reduced if is_tuple else reduced[0], chunk_size))
    if any((_num_samples(r) != chunk_size for r in reduced)):
        actual_size = tuple((_num_samples(r) for r in reduced))
        raise ValueError('reduce_func returned object of length %s. Expected same length as input: %d.' % (actual_size if is_tuple else actual_size[0], chunk_size))