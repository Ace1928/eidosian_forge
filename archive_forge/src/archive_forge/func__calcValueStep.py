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
def _calcValueStep(self):
    """Calculate _valueStep for the axis or get from valueStep."""
    if self.valueStep is None:
        rawRange = self._valueMax - self._valueMin
        rawInterval = rawRange / min(float(self.maximumTicks - 1), float(self._length) / self.minimumTickSpacing)
        if rawInterval >= self._dc:
            d = self._dc
            self._unit = 'days'
        elif rawInterval >= self._hc:
            d = self._hc
            self._unit = 'hours'
        elif rawInterval >= self._mc:
            d = self._mc
            self._unit = 'minutes'
        else:
            d = 1
            self._unit = 'seconds'
        self._unitd = d
        if d > 1:
            rawInterval = int(rawInterval / d)
        self._valueStep = nextRoundNumber(rawInterval) * d
    else:
        self._valueStep = self.valueStep