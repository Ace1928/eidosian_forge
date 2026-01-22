from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit
from numba import cuda
@ngjit
@self.expand_aggs_and_cols(append)
def _perform_extend_points(i, j, sx, tx, sy, ty, xmin, xmax, ymin, ymax, values, *aggs_and_cols):
    x = values[j]
    y = values[j + 1]
    if xmin <= x <= xmax and ymin <= y <= ymax:
        xx = int(x_mapper(x) * sx + tx)
        yy = int(y_mapper(y) * sy + ty)
        xi, yi = (xx - 1 if x == xmax else xx, yy - 1 if y == ymax else yy)
        append(i, xi, yi, *aggs_and_cols)