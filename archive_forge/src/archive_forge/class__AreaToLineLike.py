from __future__ import annotations
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.glyphs.line import _build_map_onto_pixel_for_line, _clipt
from datashader.glyphs.points import _PointLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
class _AreaToLineLike(Glyph):
    """Shared methods between Point and Line"""

    def __init__(self, x, y, y_stack):
        self.x = x
        self.y = y
        self.y_stack = y_stack

    @property
    def ndims(self):
        return 1

    @property
    def inputs(self):
        return (self.x, self.y, self.y_stack)

    def validate(self, in_dshape):
        if not isreal(in_dshape.measure[str(self.x)]):
            raise ValueError('x must be real')
        elif not isreal(in_dshape.measure[str(self.y)]):
            raise ValueError('y must be real')
        elif not isreal(in_dshape.measure[str(self.y_stack)]):
            raise ValueError('y_stack must be real or None')

    @property
    def x_label(self):
        return self.x

    @property
    def y_label(self):
        return self.y

    def required_columns(self):
        return (self.x, self.y, self.y_stack)

    def compute_x_bounds(self, df):
        bounds = self._compute_bounds(df[self.x])
        return self.maybe_expand_bounds(bounds)