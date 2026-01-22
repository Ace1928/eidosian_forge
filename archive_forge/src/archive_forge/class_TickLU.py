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
class TickLU:
    """lookup special cases for tick values"""

    def __init__(self, *T, **kwds):
        self.accuracy = kwds.pop('accuracy', 1e-08)
        self.T = T

    def __contains__(self, t):
        accuracy = self.accuracy
        for x, v in self.T:
            if abs(x - t) < accuracy:
                return True
        return False

    def __getitem__(self, t):
        accuracy = self.accuracy
        for x, v in self.T:
            if abs(x - t) < self.accuracy:
                return v
        raise IndexError('cannot locate index %r' % t)