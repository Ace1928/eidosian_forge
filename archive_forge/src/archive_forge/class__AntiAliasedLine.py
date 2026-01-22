from __future__ import annotations
import math
import numpy as np
from toolz import memoize
from datashader.antialias import two_stage_agg
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.utils import isnull, isreal, ngjit
from numba import cuda
import numba.types as nb_types
class _AntiAliasedLine:
    """ Methods common to all lines. """
    _line_width = 0

    def set_line_width(self, line_width):
        self._line_width = line_width
        if hasattr(self, 'antialiased'):
            self.antialiased = line_width > 0

    def _build_extend(self, x_mapper, y_mapper, info, append, antialias_stage_2, antialias_stage_2_funcs):
        return self._internal_build_extend(x_mapper, y_mapper, info, append, self._line_width, antialias_stage_2, antialias_stage_2_funcs)