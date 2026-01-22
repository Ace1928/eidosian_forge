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
class XLabel(Label):
    """like label but uses XPreFormatted/Paragraph to draw the _text"""
    _attrMap = AttrMap(BASE=Label)

    def __init__(self, *args, **kwds):
        Label.__init__(self, *args, **kwds)
        self.ddfKlass = kwds.pop('ddfKlass', XPreformatted)
        self.ddf = kwds.pop('directDrawClass', self.ddf)
    if False:

        def __init__(self, *args, **kwds):
            self._flowableClass = kwds.pop('flowableClass', XPreformatted)
            ddf = kwds.pop('directDrawClass', DirectDrawFlowable)
            if ddf is None:
                raise RuntimeError('DirectDrawFlowable class is not available you need the rlextra package as well as reportlab')
            self._ddf = ddf
            Label.__init__(self, *args, **kwds)

        def computeSize(self):
            self._lineWidths = []
            sty = self._style = ParagraphStyle('xlabel-generated', fontName=self.fontName, fontSize=self.fontSize, fillColor=self.fillColor, strokeColor=self.strokeColor)
            self._getBaseLineRatio()
            if self.useAscentDescent:
                sty.autoLeading = True
                sty.leading = self._ascent - self._descent
            else:
                sty.leading = self.leading if self.leading else self.fontSize * 1.2
            self._leading = sty.leading
            ta = self._getTextAnchor()
            aW = self.maxWidth or 2147483647
            if ta != 'start':
                sty.alignment = TA_LEFT
                obj = self._flowableClass(self._text, style=sty)
                _, objH = obj.wrap(aW, 2147483647)
                aW = self.maxWidth or obj._width_max
            sty.alignment = _ta2al[ta]
            self._obj = obj = self._flowableClass(self._text, style=sty)
            _, objH = obj.wrap(aW, 2147483647)
            if not self.width:
                self._width = self.leftPadding + self.rightPadding
                self._width += self._obj._width_max
            else:
                self._width = self.width
            self._computeSizeEnd(objH)

        def _rawDraw(self):
            _text = self._text
            self._text = _text or ''
            self.computeSize()
            self._text = _text
            g = Group()
            g.translate(self.x + self.dx, self.y + self.dy)
            g.rotate(self.angle)
            x = self._left
            if self.boxFillColor or (self.boxStrokeColor and self.boxStrokeWidth):
                g.add(Rect(self._left - self.leftPadding, self._bottom - self.bottomPadding, self._width, self._height, strokeColor=self.boxStrokeColor, strokeWidth=self.boxStrokeWidth, fillColor=self.boxFillColor))
            g1 = Group()
            g1.translate(x, self._top - self._eheight)
            g1.add(self._ddf(self._obj))
            g.add(g1)
            return g