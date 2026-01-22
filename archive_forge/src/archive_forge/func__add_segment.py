import logging
from fontTools.pens.basePen import AbstractPen
from fontTools.pens.pointPen import PointToSegmentPen
from fontTools.pens.reverseContourPen import ReverseContourPen
from . import curves_to_quadratic
from .errors import (
def _add_segment(self, tag, *args):
    if tag in ['move', 'line', 'qcurve', 'curve']:
        self._last_pt = args[-1]
    self.segments.append((tag, args))