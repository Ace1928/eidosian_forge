import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
def _compute_bounds_from_1d_centers(self, xr_ds, dim, maybe_expand=False, orient=True):
    vals = xr_ds[dim].values
    v0, v1, v_nm1, v_n = [vals[i] for i in [0, 1, -2, -1]]
    if v_n < v0:
        descending = True
        v0, v1, v_nm1, v_n = (v_n, v_nm1, v1, v0)
    else:
        descending = False
    bounds = (v0 - 0.5 * (v1 - v0), v_n + 0.5 * (v_n - v_nm1))
    if not orient and descending:
        bounds = (bounds[1], bounds[0])
    if maybe_expand:
        bounds = self.maybe_expand_bounds(bounds)
    return bounds