from __future__ import annotations
import math
from collections import namedtuple
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import FlexBox as BkFlexBox, GridBox as BkGridBox
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from .base import (
@property
def _object_grid(self):
    grid = np.full((self.nrows, self.ncols), None, dtype=object)
    for (y0, x0, y1, x1), obj in self.objects.items():
        l = 0 if x0 is None else x0
        r = self.ncols if x1 is None else x1
        t = 0 if y0 is None else y0
        b = self.nrows if y1 is None else y1
        for y in range(t, b):
            for x in range(l, r):
                grid[y, x] = {((y0, x0, y1, x1), obj)}
    return grid