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
class LogAxisLabellingSetup:

    def __init__(self):
        if DirectDrawFlowable is not None:
            self.labels = TypedPropertyCollection(XLabel)
            if self._dataIndex == 1:
                self.labels.boxAnchor = 'e'
                self.labels.dx = -5
                self.labels.dy = 0
            else:
                self.labels.boxAnchor = 'n'
                self.labels.dx = 0
                self.labels.dy = -5
            self.labelTextFormat = LogAxisTickLabeller()
        else:
            self.labelTextFormat = LogAxisTickLabellerS()