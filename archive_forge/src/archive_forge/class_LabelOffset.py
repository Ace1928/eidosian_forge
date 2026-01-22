from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
from reportlab.lib.validators import isNumber, isNumberOrNone, OneOf, isColorOrNone, isString, \
from reportlab.lib.attrmap import *
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.graphics.shapes import Drawing, Group, Circle, Rect, String, STATE_DEFAULTS
from reportlab.graphics.widgetbase import Widget, PropHolder
from reportlab.graphics.shapes import DirectDraw
from reportlab.platypus import XPreformatted, Flowable
from reportlab.lib.styles import ParagraphStyle, PropertySet
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
from ..utils import text2Path as _text2Path   #here for continuity
from reportlab.graphics.charts.utils import CustomDrawChanger
class LabelOffset(PropHolder):
    _attrMap = AttrMap(posMode=AttrMapValue(isOffsetMode, desc='Where to base +ve offset'), pos=AttrMapValue(isNumber, desc='Value for positive elements'), negMode=AttrMapValue(isOffsetMode, desc='Where to base -ve offset'), neg=AttrMapValue(isNumber, desc='Value for negative elements'))

    def __init__(self):
        self.posMode = self.negMode = 'axis'
        self.pos = self.neg = 0

    def _getValue(self, chart, val):
        flipXY = chart._flipXY
        A = chart.categoryAxis
        jA = A.joinAxis
        if val >= 0:
            mode = self.posMode
            delta = self.pos
        else:
            mode = self.negMode
            delta = self.neg
        if flipXY:
            v = A._x
        else:
            v = A._y
        if jA:
            if flipXY:
                _v = jA._x
            else:
                _v = jA._y
            if mode == 'high':
                v = _v + jA._length
            elif mode == 'low':
                v = _v
            elif mode == 'bar':
                v = _v + val
        return v + delta