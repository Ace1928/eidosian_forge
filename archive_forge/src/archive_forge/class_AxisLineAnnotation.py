from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
class AxisLineAnnotation:
    """Create a grid like line using the given user value to draw the line
    kwds may contain
    startOffset if true v is offset from the default grid start position
    endOffset   if true v is offset from the default grid end position
    scaleValue  True/not given --> scale the value
                otherwise use the absolute value
    lo          lowest coordinate to draw default 0
    hi          highest coordinate to draw at default = length
    drawAtLimit True draw line at appropriate limit if its coordinate exceeds the lo, hi range
                False ignore if it's outside the range
    all Line keywords are acceptable
    """

    def __init__(self, v, **kwds):
        self._v = v
        self._kwds = kwds

    def __call__(self, axis):
        kwds = self._kwds.copy()
        scaleValue = kwds.pop('scaleValue', True)
        endOffset = kwds.pop('endOffset', False)
        startOffset = kwds.pop('startOffset', False)
        if axis.isYAxis:
            offs = axis._x
            d0 = axis._y
        else:
            offs = axis._y
            d0 = axis._x
        s = kwds.pop('start', None)
        e = kwds.pop('end', None)
        if s is None or e is None:
            dim = getattr(getattr(axis, 'joinAxis', None), 'getGridDims', None)
            if dim and hasattr(dim, '__call__'):
                dim = dim()
            if dim:
                if s is None:
                    s = dim[0]
                if e is None:
                    e = dim[1]
            else:
                if s is None:
                    s = 0
                if e is None:
                    e = 0
        hi = kwds.pop('hi', axis._length) + d0
        lo = kwds.pop('lo', 0) + d0
        lo, hi = (min(lo, hi), max(lo, hi))
        drawAtLimit = kwds.pop('drawAtLimit', False)
        oaglp = axis._get_line_pos
        if not scaleValue:
            axis._get_line_pos = lambda x: x
        try:
            v = self._v
            if endOffset:
                v = v + hi
            elif startOffset:
                v = v + lo
            func = axis._getLineFunc(s - offs, e - offs, kwds.pop('parent', None))
            if not hasattr(axis, '_tickValues'):
                axis._pseudo_configure()
            d = axis._get_line_pos(v)
            if d < lo or d > hi:
                if not drawAtLimit:
                    return None
                if d < lo:
                    d = lo
                else:
                    d = hi
                axis._get_line_pos = lambda x: d
            L = func(v)
            for k, v in kwds.items():
                setattr(L, k, v)
        finally:
            axis._get_line_pos = oaglp
        return L