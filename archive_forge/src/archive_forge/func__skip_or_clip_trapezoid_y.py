from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
@ngjit
def _skip_or_clip_trapezoid_y(x0, x1, y0, y1, y2, y3, xmin, xmax, ymin, ymax):
    skip = False
    if isnull(x0) or isnull(x1) or isnull(y0) or isnull(y1) or isnull(y2) or isnull(y3):
        skip = True
    if y0 > ymax and y1 > ymax and (y2 > ymax) and (y3 > ymax) or (y0 < ymin and y1 < ymin and (y2 < ymin) and (y3 < ymin)):
        skip = True
        clipped_start = clipped_end = False
        return (x0, x1, y0, y1, y2, y3, skip, clipped_start, clipped_end)
    t0, t1 = (0, 1)
    dx = x1 - x0
    dy0 = y3 - y0
    dy1 = y2 - y1
    t0, t1, accept = _clipt(-dx, x0 - xmin, t0, t1)
    if not accept:
        skip = True
    t0, t1, accept = _clipt(dx, xmax - x0, t0, t1)
    if not accept:
        skip = True
    if t1 < 1:
        clipped_end = True
        x1 = x0 + t1 * dx
        y2 = y1 + t1 * dy1
        y3 = y0 + t1 * dy0
    else:
        clipped_end = False
    if t0 > 0:
        clipped_start = True
        x0 = x0 + t0 * dx
        y0 = y0 + t0 * dy0
        y1 = y1 + t0 * dy1
    else:
        clipped_start = False
    return (x0, x1, y0, y1, y2, y3, skip, clipped_start, clipped_end)