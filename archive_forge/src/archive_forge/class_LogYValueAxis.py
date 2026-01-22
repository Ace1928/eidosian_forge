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
class LogYValueAxis(LogValueAxis, LogAxisLabellingSetup, YValueAxis):
    _attrMap = AttrMap(BASE=YValueAxis)

    def __init__(self):
        YValueAxis.__init__(self)
        LogAxisLabellingSetup.__init__(self)

    def scale(self, value):
        """Converts a numeric value to a Y position.

        The chart first configures the axis, then asks it to
        work out the x value for each point when plotting
        lines or bars.  You could override this to do
        logarithmic axes.
        """
        msg = 'Axis cannot scale numbers before it is configured'
        assert self._configured, msg
        if value is None:
            value = 0
        if value == 0.0:
            return self._y - self._scaleFactor * math_log10(self._valueMin)
        return self._y + self._scaleFactor * (math_log10(value) - math_log10(self._valueMin))