from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
def _build_extend_area_to_zero_axis0(draw_trapezoid_y, expand_aggs_and_cols):

    @ngjit
    @expand_aggs_and_cols
    def perform_extend(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, *aggs_and_cols):
        stacked = False
        x0 = xs[i]
        x1 = xs[i + 1]
        y0 = ys[i]
        y1 = 0.0
        y2 = 0.0
        y3 = ys[i + 1]
        trapezoid_start = plot_start if i == 0 else isnull(xs[i - 1]) or isnull(ys[i - 1])
        draw_trapezoid_y(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, x0, x1, y0, y1, y2, y3, trapezoid_start, stacked, *aggs_and_cols)

    @ngjit
    @expand_aggs_and_cols
    def extend_cpu(sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, *aggs_and_cols):
        """Aggregate filled area along a line formed by
        ``xs`` and ``ys``, filled to the y=0 line"""
        nrows = xs.shape[0]
        for i in range(nrows - 1):
            perform_extend(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, *aggs_and_cols)

    @cuda.jit
    @expand_aggs_and_cols
    def extend_cuda(sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, *aggs_and_cols):
        i = cuda.grid(1)
        if i < xs.shape[0] - 1:
            perform_extend(i, sx, tx, sy, ty, xmin, xmax, ymin, ymax, plot_start, xs, ys, *aggs_and_cols)
    return (extend_cpu, extend_cuda)