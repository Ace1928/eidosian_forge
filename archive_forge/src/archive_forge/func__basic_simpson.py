from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Any, cast
import numpy as np
import numpy.typing as npt
import math
import warnings
from collections import namedtuple
from scipy.special import roots_legendre
from scipy.special import gammaln, logsumexp
from scipy._lib._util import _rng_spawn
from scipy._lib.deprecation import (_NoValue, _deprecate_positional_args,
def _basic_simpson(y, start, stop, x, dx, axis):
    nd = len(y.shape)
    if start is None:
        start = 0
    step = 2
    slice_all = (slice(None),) * nd
    slice0 = tupleset(slice_all, axis, slice(start, stop, step))
    slice1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
    slice2 = tupleset(slice_all, axis, slice(start + 2, stop + 2, step))
    if x is None:
        result = np.sum(y[slice0] + 4.0 * y[slice1] + y[slice2], axis=axis)
        result *= dx / 3.0
    else:
        h = np.diff(x, axis=axis)
        sl0 = tupleset(slice_all, axis, slice(start, stop, step))
        sl1 = tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        h0 = h[sl0].astype(float, copy=False)
        h1 = h[sl1].astype(float, copy=False)
        hsum = h0 + h1
        hprod = h0 * h1
        h0divh1 = np.true_divide(h0, h1, out=np.zeros_like(h0), where=h1 != 0)
        tmp = hsum / 6.0 * (y[slice0] * (2.0 - np.true_divide(1.0, h0divh1, out=np.zeros_like(h0divh1), where=h0divh1 != 0)) + y[slice1] * (hsum * np.true_divide(hsum, hprod, out=np.zeros_like(hsum), where=hprod != 0)) + y[slice2] * (2.0 - h0divh1))
        result = np.sum(tmp, axis=axis)
    return result