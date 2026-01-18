import math
from toolz import memoize
import numpy as np
from datashader.glyphs.glyph import Glyph
from datashader.resampling import infer_interval_breaks
from datashader.utils import isreal, ngjit, ngjit_parallel
import numba
from numba import cuda, prange
@ngjit_parallel
def upsample_cpu(src_w, src_h, translate_x, translate_y, scale_x, scale_y, offset_x, offset_y, out_w, out_h, agg, col):
    for out_j in prange(out_h):
        src_j = int(math.floor(scale_y * (out_j + 0.5) + translate_y - offset_y))
        for out_i in range(out_w):
            src_i = int(math.floor(scale_x * (out_i + 0.5) + translate_x - offset_x))
            if src_j < 0 or src_j >= src_h or src_i < 0 or (src_i >= src_w):
                agg[out_j, out_i] = np.nan
            else:
                agg[out_j, out_i] = col[src_j, src_i]