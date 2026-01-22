from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
@ngjit_parallel
def _downsample_2d_mean(src, mask, use_mask, method, fill_value, mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    if src_w == out_w and src_h == out_h:
        return src
    if out_w > src_w or out_h > src_h:
        raise ValueError('invalid target size')
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    scale_x = (src_w - x0_off - x1_off) / out_w
    scale_y = (src_h - y0_off - y1_off) / out_h
    for out_y in prange(out_h):
        src_yf0 = scale_y * out_y + y0_off
        src_yf1 = src_yf0 + scale_y
        src_y0 = int(src_yf0)
        src_y1 = int(src_yf1)
        wy0 = 1.0 - (src_yf0 - src_y0)
        wy1 = src_yf1 - src_y1
        if wy1 < _EPS:
            wy1 = 1.0
            if src_y1 > src_y0:
                src_y1 -= 1
        for out_x in range(out_w):
            src_xf0 = scale_x * out_x + x0_off
            src_xf1 = src_xf0 + scale_x
            src_x0 = int(src_xf0)
            src_x1 = int(src_xf1)
            wx0 = 1.0 - (src_xf0 - src_x0)
            wx1 = src_xf1 - src_x1
            if wx1 < _EPS:
                wx1 = 1.0
                if src_x1 > src_x0:
                    src_x1 -= 1
            v_sum = 0.0
            w_sum = 0.0
            for src_y in range(src_y0, src_y1 + 1):
                wy = wy0 if src_y == src_y0 else wy1 if src_y == src_y1 else 1.0
                for src_x in range(src_x0, src_x1 + 1):
                    wx = wx0 if src_x == src_x0 else wx1 if src_x == src_x1 else 1.0
                    v = src[src_y, src_x]
                    if np.isfinite(v) and (not (use_mask and mask[src_y, src_x])):
                        w = wx * wy
                        v_sum += w * v
                        w_sum += w
            if w_sum < _EPS:
                out[out_y, out_x] = fill_value
            else:
                out[out_y, out_x] = v_sum / w_sum
    return out