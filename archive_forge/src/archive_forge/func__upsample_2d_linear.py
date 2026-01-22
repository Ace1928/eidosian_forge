from __future__ import annotations
from itertools import groupby
from math import floor, ceil
import dask.array as da
import numpy as np
from dask.delayed import delayed
from numba import prange
from .utils import ngjit, ngjit_parallel
@ngjit_parallel
def _upsample_2d_linear(src, mask, use_mask, fill_value, x_offset, y_offset, out):
    src_w, src_h, out_w, out_h = _get_dimensions(src, out)
    x0_off, x1_off = x_offset
    y0_off, y1_off = y_offset
    src_wo = src_w - x0_off - x1_off
    src_ho = src_h - y0_off - y1_off
    if src_wo == out_w and src_ho == out_h:
        return src
    if out_w < src_w or out_h < src_h:
        raise ValueError('invalid target size')
    scale_x = (src_wo - 1.0) / (out_w - 1.0 if out_w > 1 else 1.0)
    scale_y = (src_ho - 1.0) / (out_h - 1.0 if out_h > 1 else 1.0)
    for out_y in prange(out_h):
        src_yf = scale_y * out_y + y0_off
        src_y0 = int(src_yf)
        wy = src_yf - src_y0
        src_y1 = src_y0 + 1
        if src_y1 >= src_h:
            src_y1 = src_y0
        for out_x in range(out_w):
            src_xf = scale_x * out_x + x0_off
            src_x0 = int(src_xf)
            wx = src_xf - src_x0
            src_x1 = src_x0 + 1
            if src_x1 >= src_w:
                src_x1 = src_x0
            v00 = src[src_y0, src_x0]
            v01 = src[src_y0, src_x1]
            v10 = src[src_y1, src_x0]
            v11 = src[src_y1, src_x1]
            if use_mask:
                v00_ok = np.isfinite(v00) and (not mask[src_y0, src_x0])
                v01_ok = np.isfinite(v01) and (not mask[src_y0, src_x1])
                v10_ok = np.isfinite(v10) and (not mask[src_y1, src_x0])
                v11_ok = np.isfinite(v11) and (not mask[src_y1, src_x1])
            else:
                v00_ok = np.isfinite(v00)
                v01_ok = np.isfinite(v01)
                v10_ok = np.isfinite(v10)
                v11_ok = np.isfinite(v11)
            if v00_ok and v01_ok and v10_ok and v11_ok:
                ok = True
                v0 = v00 + wx * (v01 - v00)
                v1 = v10 + wx * (v11 - v10)
                value = v0 + wy * (v1 - v0)
            elif wx < 0.5:
                if wy < 0.5:
                    ok = v00_ok
                    value = v00
                else:
                    ok = v10_ok
                    value = v10
            elif wy < 0.5:
                ok = v01_ok
                value = v01
            else:
                ok = v11_ok
                value = v11
            if ok:
                out[out_y, out_x] = value
            else:
                out[out_y, out_x] = fill_value
    return out