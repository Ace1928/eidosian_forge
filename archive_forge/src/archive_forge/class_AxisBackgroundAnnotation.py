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
class AxisBackgroundAnnotation:
    """Create a set of coloured bars on the background of a chart using axis ticks as the bar borders
    colors is a set of colors to use for the background bars. A colour of None is just a skip.
    Special effects if you pass a rect or Shaded rect instead.
    """

    def __init__(self, colors, **kwds):
        self._colors = colors
        self._kwds = kwds

    def __call__(self, axis):
        colors = self._colors
        if not colors:
            return
        kwds = self._kwds.copy()
        isYAxis = axis.isYAxis
        if isYAxis:
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
        if not hasattr(axis, '_tickValues'):
            axis._pseudo_configure()
        tv = getattr(axis, '_tickValues', None)
        if not tv:
            return
        G = Group()
        ncolors = len(colors)
        v0 = axis._get_line_pos(tv[0])
        for i in range(1, len(tv)):
            v1 = axis._get_line_pos(tv[i])
            c = colors[(i - 1) % ncolors]
            if c:
                if isYAxis:
                    y = v0
                    x = s
                    height = v1 - v0
                    width = e - s
                else:
                    x = v0
                    y = s
                    width = v1 - v0
                    height = e - s
                if isinstance(c, Color):
                    r = Rect(x, y, width, height, fillColor=c, strokeColor=None)
                elif isinstance(c, Rect):
                    r = Rect(x, y, width, height)
                    for k in c.__dict__:
                        if k not in ('x', 'y', 'width', 'height'):
                            setattr(r, k, getattr(c, k))
                elif isinstance(c, ShadedRect):
                    r = ShadedRect(x=x, y=y, width=width, height=height)
                    for k in c.__dict__:
                        if k not in ('x', 'y', 'width', 'height'):
                            setattr(r, k, getattr(c, k))
                G.add(r)
            v0 = v1
        return G