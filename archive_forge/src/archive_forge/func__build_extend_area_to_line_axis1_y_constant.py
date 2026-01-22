from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
def _build_extend_area_to_line_axis1_y_constant(draw_trapezoid_y, expand_aggs_and_cols):

    @ngjit
    @expand_aggs_and_cols
    def perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys0, ys1, *aggs_and_cols):
        x0 = xs[i, j]
        x1 = xs[i, j + 1]
        y0 = ys0[j]
        y1 = ys1[j]
        y2 = ys1[j + 1]
        y3 = ys0[j + 1]
        trapezoid_start = j == 0 or isnull(xs[i, j - 1]) or isnull(ys0[j - 1]) or isnull(ys1[j - 1])
        stacked = True
        draw_trapezoid_y(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys0, ys1, *aggs_and_cols):
        nrows, ncols = xs.shape
        for i in range(nrows):
            for j in range(ncols - 1):
                perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys0, ys1, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys0, ys1, *aggs_and_cols):
        i, j = cuda.grid(2)
        if i < xs.shape[0] and j < xs.shape[1] - 1:
            perform_extend(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, xs, ys0, ys1, *aggs_and_cols)
    return (extend_cpu, extend_cuda)