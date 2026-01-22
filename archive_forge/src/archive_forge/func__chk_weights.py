import sys
import os.path
from functools import wraps, partial
import weakref
import numpy as np
import warnings
from numpy.linalg import norm
from numpy.testing import (verbose, assert_,
import pytest
import scipy.spatial.distance
from scipy.spatial.distance import (
from scipy.spatial.distance import (braycurtis, canberra, chebyshev, cityblock,
from scipy._lib._util import np_long, np_ulong
def _chk_weights(arrays, weights=None, axis=None, force_weights=False, simplify_weights=True, pos_only=False, neg_check=False, nan_screen=False, mask_screen=False, ddof=None):
    chked = _chk_asarrays(arrays, axis=axis)
    arrays, axis = (chked[:-1], chked[-1])
    simplify_weights = simplify_weights and (not force_weights)
    if not force_weights and mask_screen:
        force_weights = any((np.ma.getmask(a) is not np.ma.nomask for a in arrays))
    if nan_screen:
        has_nans = [np.isnan(np.sum(a)) for a in arrays]
        if any(has_nans):
            mask_screen = True
            force_weights = True
            arrays = tuple((np.ma.masked_invalid(a) if has_nan else a for a, has_nan in zip(arrays, has_nans)))
    if weights is not None:
        weights = np.asanyarray(weights)
    elif force_weights:
        weights = np.ones(arrays[0].shape[axis])
    else:
        return arrays + (weights, axis)
    if ddof:
        weights = _freq_weights(weights)
    if mask_screen:
        weights = _weight_masked(arrays, weights, axis)
    if not all((weights.shape == (a.shape[axis],) for a in arrays)):
        raise ValueError('weights shape must match arrays along axis')
    if neg_check and (weights < 0).any():
        raise ValueError('weights cannot be negative')
    if pos_only:
        pos_weights = np.nonzero(weights > 0)[0]
        if pos_weights.size < weights.size:
            arrays = tuple((np.take(a, pos_weights, axis=axis) for a in arrays))
            weights = weights[pos_weights]
    if simplify_weights and (weights == 1).all():
        weights = None
    return arrays + (weights, axis)