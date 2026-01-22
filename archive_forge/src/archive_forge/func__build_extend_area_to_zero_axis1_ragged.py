from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
def _build_extend_area_to_zero_axis1_ragged(draw_trapezoid_y, expand_aggs_and_cols):

    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys, *aggs_and_cols):
        x_start_inds = xs.start_indices
        x_flat = xs.flat_array
        y_start_inds = ys.start_indices
        y_flat = ys.flat_array
        perform_extend_area_to_zero_axis1_ragged(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_inds, x_flat, y_start_inds, y_flat, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def perform_extend_area_to_zero_axis1_ragged(sx, tx, sy, ty, xmin, xmax, ymin, ymax, x_start_inds, x_flat, y_start_inds, y_flat, *aggs_and_cols):
        nrows = len(x_start_inds)
        x_flat_len = len(x_flat)
        y_flat_len = len(y_flat)
        i = 0
        while i < nrows:
            x_start_i = x_start_inds[i]
            x_stop_i = x_start_inds[i + 1] if i < nrows - 1 else x_flat_len
            y_start_i = y_start_inds[i]
            y_stop_i = y_start_inds[i + 1] if i < nrows - 1 else y_flat_len
            segment_len = min(x_stop_i - x_start_i, y_stop_i - y_start_i)
            j = 0
            while j < segment_len - 1:
                x0 = x_flat[x_start_i + j]
                x1 = x_flat[x_start_i + j + 1]
                y0 = y_flat[y_start_i + j]
                y1 = 0.0
                y2 = 0.0
                y3 = y_flat[y_start_i + j + 1]
                trapezoid_start = j == 0 or isnull(x_flat[x_start_i + j - 1]) or isnull(y_flat[y_start_i + j - 1])
                stacked = False
                draw_trapezoid_y(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols)
                j += 1
            i += 1
    return extend_cpu