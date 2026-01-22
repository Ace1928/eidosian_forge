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
class LabelDecorator:
    _attrMap = AttrMap(x=AttrMapValue(isNumberOrNone, desc=''), y=AttrMapValue(isNumberOrNone, desc=''), dx=AttrMapValue(isNumberOrNone, desc=''), dy=AttrMapValue(isNumberOrNone, desc=''), angle=AttrMapValue(isNumberOrNone, desc=''), boxAnchor=AttrMapValue(isBoxAnchor, desc=''), boxStrokeColor=AttrMapValue(isColorOrNone, desc=''), boxStrokeWidth=AttrMapValue(isNumberOrNone, desc=''), boxFillColor=AttrMapValue(isColorOrNone, desc=''), fillColor=AttrMapValue(isColorOrNone, desc=''), strokeColor=AttrMapValue(isColorOrNone, desc=''), strokeWidth=AttrMapValue(isNumberOrNone), desc='', fontName=AttrMapValue(isNoneOrString, desc=''), fontSize=AttrMapValue(isNumberOrNone, desc=''), leading=AttrMapValue(isNumberOrNone, desc=''), width=AttrMapValue(isNumberOrNone, desc=''), maxWidth=AttrMapValue(isNumberOrNone, desc=''), height=AttrMapValue(isNumberOrNone, desc=''), textAnchor=AttrMapValue(isTextAnchor, desc=''), visible=AttrMapValue(isBoolean, desc='True if the label is to be drawn'))

    def __init__(self):
        self.textAnchor = 'start'
        self.boxAnchor = 'w'
        for a in self._attrMap.keys():
            if not hasattr(self, a):
                setattr(self, a, None)

    def decorate(self, l, L):
        chart, g, rowNo, colNo, x, y, width, height, x00, y00, x0, y0 = l._callOutInfo
        L.setText(chart.categoryAxis.categoryNames[colNo])
        g.add(L)

    def __call__(self, l):
        L = Label()
        for a, v in self.__dict__.items():
            if v is None:
                v = getattr(l, a, None)
            setattr(L, a, v)
        self.decorate(l, L)