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
def _getValueStepAndTicks(self, valueMin, valueMax, cache={}):
    try:
        K = (valueMin, valueMax)
        r = cache[K]
    except:
        self._valueMin = valueMin
        self._valueMax = valueMax
        valueStep, T = self._calcStepAndTickPositions()
        r = cache[K] = (valueStep, T, valueStep * 1e-08)
    return r