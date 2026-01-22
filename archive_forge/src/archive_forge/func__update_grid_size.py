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
@param.depends('objects', watch=True)
def _update_grid_size(self):
    self._updating = True
    if not self._cols_fixed:
        max_xidx = [x1 for _, _, _, x1 in self.objects if x1 is not None]
        self.ncols = max(max_xidx) if max_xidx else 1 if len(self.objects) else 0
    if not self._rows_fixed:
        max_yidx = [y1 for _, _, y1, _ in self.objects if y1 is not None]
        self.nrows = max(max_yidx) if max_yidx else 1 if len(self.objects) else 0
    self._updating = False