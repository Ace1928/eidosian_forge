from itertools import product
from numbers import Integral, Number, Real
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from ..base import BaseEstimator, TransformerMixin, _fit_context
from ..utils import check_array, check_random_state
from ..utils._param_validation import Hidden, Interval, RealNotInt, validate_params
def _mask_edges_weights(mask, edges, weights=None):
    """Apply a mask to edges (weighted or not)"""
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.isin(edges[0], inds), np.isin(edges[1], inds))
    edges = edges[:, ind_mask]
    if weights is not None:
        weights = weights[ind_mask]
    if len(edges.ravel()):
        maxval = edges.max()
    else:
        maxval = 0
    order = np.searchsorted(np.flatnonzero(mask), np.arange(maxval + 1))
    edges = order[edges]
    if weights is None:
        return edges
    else:
        return (edges, weights)