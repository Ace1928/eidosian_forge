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
def _makeLines(self, g, start, end, strokeColor, strokeWidth, strokeDashArray, strokeLineJoin, strokeLineCap, strokeMiterLimit, parent=None, exclude=[], specials={}):
    func = self._getLineFunc(start, end, parent)
    if not hasattr(self, '_tickValues'):
        self._pseudo_configure()
    if exclude:
        exf = self.isYAxis and (lambda l: l.y1 in exclude) or (lambda l: l.x1 in exclude)
    else:
        exf = None
    for t in self._tickValues:
        L = func(t)
        if exf and exf(L):
            continue
        L.strokeColor = strokeColor
        L.strokeWidth = strokeWidth
        L.strokeDashArray = strokeDashArray
        L.strokeLineJoin = strokeLineJoin
        L.strokeLineCap = strokeLineCap
        L.strokeMiterLimit = strokeMiterLimit
        if t in specials:
            for a, v in specials[t].items():
                setattr(L, a, v)
        g.add(L)