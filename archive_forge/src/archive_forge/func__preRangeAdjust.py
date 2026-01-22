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
def _preRangeAdjust(self, valueMin, valueMax):
    rr = self.requiredRange
    if rr > 0:
        r = valueMax - valueMin
        if r < rr:
            m = 0.5 * (valueMax + valueMin)
            rr *= 0.5
            y1 = min(m - rr, valueMin)
            y2 = max(m + rr, valueMax)
            if valueMin >= 100 and y1 < 100:
                y2 = y2 + 100 - y1
                y1 = 100
            elif valueMin >= 0 and y1 < 0:
                y2 = y2 - y1
                y1 = 0
            valueMin = self._cValueMin = y1
            valueMax = self._cValueMax = y2
    return (valueMin, valueMax)