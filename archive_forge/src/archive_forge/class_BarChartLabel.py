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
class BarChartLabel(PMVLabel):
    """
    An extended Label allowing for nudging, lines visibility etc
    """
    _attrMap = AttrMap(BASE=PMVLabel, lineStrokeWidth=AttrMapValue(isNumberOrNone, desc='Non-zero for a drawn line'), lineStrokeColor=AttrMapValue(isColorOrNone, desc='Color for a drawn line'), fixedEnd=AttrMapValue(NoneOrInstanceOfLabelOffset, desc='None or fixed draw ends +/-'), fixedStart=AttrMapValue(NoneOrInstanceOfLabelOffset, desc='None or fixed draw starts +/-'), nudge=AttrMapValue(isNumber, desc='Non-zero sign dependent nudge'), boxTarget=AttrMapValue(OneOf('normal', 'anti', 'lo', 'hi', 'mid'), desc="one of ('normal','anti','lo','hi','mid')"))

    def __init__(self, **kwds):
        PMVLabel.__init__(self, **kwds)
        self.lineStrokeWidth = 0
        self.lineStrokeColor = None
        self.fixedStart = self.fixedEnd = None
        self.nudge = 0