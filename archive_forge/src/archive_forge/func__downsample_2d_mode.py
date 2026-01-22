from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
@ngjit_parallel
def _downsample_2d_mode(src, mask, use_mask, method, fill_value, mode_rank, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    if src_w == out_w and src_h == out_h:
        return src
    if out_w > src_w or out_h > src_h:
        raise ValueError('invalid target size')
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    scale_x = (src_w - x0_off - x1_off) / out_w
    scale_y = (src_h - y0_off - y1_off) / out_h
    max_value_count = ceil(scale_x + 1) * ceil(scale_y + 1)
    if mode_rank >= max_value_count:
        raise ValueError('requested mode_rank too large for max_value_count being collected')
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
            values = np.zeros((max_value_count,), dtype=src.dtype)
            frequencies = np.zeros((max_value_count,), dtype=np.uint32)
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
            value_count = 0
            for src_y in range(src_y0, src_y1 + 1):
                wy = wy0 if src_y == src_y0 else wy1 if src_y == src_y1 else 1.0
                for src_x in range(src_x0, src_x1 + 1):
                    wx = wx0 if src_x == src_x0 else wx1 if src_x == src_x1 else 1.0
                    v = src[src_y, src_x]
                    if np.isfinite(v) and (not (use_mask and mask[src_y, src_x])):
                        w = wx * wy
                        found = False
                        for i in range(value_count):
                            if v == values[i]:
                                frequencies[i] += w
                                found = True
                                break
                        if not found:
                            values[value_count] = v
                            frequencies[value_count] = w
                            value_count += 1
            w_max = -1.0
            value = fill_value
            if mode_rank == 1:
                for i in range(value_count):
                    w = frequencies[i]
                    if w > w_max:
                        w_max = w
                        value = values[i]
            elif mode_rank <= max_value_count:
                max_frequencies = np.full(mode_rank, -1.0, dtype=np.float64)
                indices = np.zeros(mode_rank, dtype=np.int64)
                for i in range(value_count):
                    w = frequencies[i]
                    for j in range(mode_rank):
                        if w > max_frequencies[j]:
                            max_frequencies[j] = w
                            indices[j] = i
                            break
                value = values[indices[mode_rank - 1]]
            out[out_y, out_x] = value
    return out