import warnings
from numbers import Integral, Real
import numpy as np
from scipy.sparse import SparseEfficiencyWarning, issparse
from ..base import BaseEstimator, ClusterMixin, _fit_context
from ..exceptions import DataConversionWarning
from ..metrics import pairwise_distances
from ..metrics.pairwise import _VALID_METRICS, PAIRWISE_BOOLEAN_FUNCTIONS
from ..neighbors import NearestNeighbors
from ..utils import gen_batches, get_chunk_n_rows
from ..utils._param_validation import (
from ..utils.validation import check_memory
def _set_reach_dist(core_distances_, reachability_, predecessor_, point_index, processed, X, nbrs, metric, metric_params, p, max_eps):
    P = X[point_index:point_index + 1]
    indices = nbrs.radius_neighbors(P, radius=max_eps, return_distance=False)[0]
    unproc = np.compress(~np.take(processed, indices), indices)
    if not unproc.size:
        return
    if metric == 'precomputed':
        dists = X[[point_index], unproc]
        if isinstance(dists, np.matrix):
            dists = np.asarray(dists)
        dists = dists.ravel()
    else:
        _params = dict() if metric_params is None else metric_params.copy()
        if metric == 'minkowski' and 'p' not in _params:
            _params['p'] = p
        dists = pairwise_distances(P, X[unproc], metric, n_jobs=None, **_params).ravel()
    rdists = np.maximum(dists, core_distances_[point_index])
    np.around(rdists, decimals=np.finfo(rdists.dtype).precision, out=rdists)
    improved = np.where(rdists < np.take(reachability_, unproc))
    reachability_[unproc[improved]] = rdists[improved]
    predecessor_[unproc[improved]] = point_index