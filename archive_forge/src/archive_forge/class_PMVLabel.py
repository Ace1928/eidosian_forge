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
class PMVLabel(Label):
    _attrMap = AttrMap(BASE=Label)

    def __init__(self, **kwds):
        Label.__init__(self, **kwds)
        self._pmv = 0

    def _getBoxAnchor(self):
        a = Label._getBoxAnchor(self)
        if self._pmv < 0:
            a = {'nw': 'se', 'n': 's', 'ne': 'sw', 'w': 'e', 'c': 'c', 'e': 'w', 'sw': 'ne', 's': 'n', 'se': 'nw'}[a]
        return a

    def _getTextAnchor(self):
        a = Label._getTextAnchor(self)
        if self._pmv < 0:
            a = {'start': 'end', 'middle': 'middle', 'end': 'start'}[a]
        return a