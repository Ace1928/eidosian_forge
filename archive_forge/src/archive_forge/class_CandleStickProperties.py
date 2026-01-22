from reportlab.graphics import shapes
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from weakref import ref as weakref_ref
class CandleStickProperties(PropHolder):
    _attrMap = AttrMap(strokeWidth=AttrMapValue(isNumber, desc='Width of a line.'), strokeColor=AttrMapValue(isColorOrNone, desc='Color of a line or border.'), strokeDashArray=AttrMapValue(isListOfNumbersOrNone, desc='Dash array of a line.'), crossWidth=AttrMapValue(isNumberOrNone, desc='cross line width', advancedUsage=1), crossLo=AttrMapValue(isNumberOrNone, desc='cross line low value', advancedUsage=1), crossHi=AttrMapValue(isNumberOrNone, desc='cross line high value', advancedUsage=1), boxWidth=AttrMapValue(isNumberOrNone, desc='width of the box part', advancedUsage=1), boxFillColor=AttrMapValue(isColorOrNone, desc='fill color of box'), boxStrokeColor=AttrMapValue(NotSetOr(isColorOrNone), desc='stroke color of box'), boxStrokeDashArray=AttrMapValue(NotSetOr(isListOfNumbersOrNone), desc='Dash array of the box.'), boxStrokeWidth=AttrMapValue(NotSetOr(isNumber), desc='Width of the box lines.'), boxLo=AttrMapValue(isNumberOrNone, desc='low value of the box', advancedUsage=1), boxMid=AttrMapValue(isNumberOrNone, desc='middle box line value', advancedUsage=1), boxHi=AttrMapValue(isNumberOrNone, desc='high value of the box', advancedUsage=1), boxSides=AttrMapValue(isBoolean, desc='whether to show box sides', advancedUsage=1), position=AttrMapValue(isNumberOrNone, desc='position of the candle', advancedUsage=1), chart=AttrMapValue(None, desc='our chart', advancedUsage=1), candleKind=AttrMapValue(OneOf('vertical', 'horizontal'), desc='candle direction', advancedUsage=1), axes=AttrMapValue(SequenceOf(isString, emptyOK=0, lo=2, hi=2), desc='candle direction', advancedUsage=1))

    def __init__(self, **kwds):
        self.strokeWidth = kwds.pop('strokeWidth', 1)
        self.strokeColor = kwds.pop('strokeColor', colors.black)
        self.strokeDashArray = kwds.pop('strokeDashArray', None)
        self.crossWidth = kwds.pop('crossWidth', 5)
        self.crossLo = kwds.pop('crossLo', None)
        self.crossHi = kwds.pop('crossHi', None)
        self.boxWidth = kwds.pop('boxWidth', None)
        self.boxFillColor = kwds.pop('boxFillColor', None)
        self.boxStrokeColor = kwds.pop('boxStrokeColor', NotSetOr._not_set)
        self.boxStrokeWidth = kwds.pop('boxStrokeWidth', NotSetOr._not_set)
        self.boxStrokeDashArray = kwds.pop('boxStrokeDashArray', NotSetOr._not_set)
        self.boxLo = kwds.pop('boxLo', None)
        self.boxMid = kwds.pop('boxMid', None)
        self.boxHi = kwds.pop('boxHi', None)
        self.boxSides = kwds.pop('boxSides', True)
        self.position = kwds.pop('position', None)
        self.candleKind = kwds.pop('candleKind', 'vertical')
        self.axes = kwds.pop('axes', ['categoryAxis', 'valueAxis'])
        chart = kwds.pop('chart', None)
        self.chart = weakref_ref(chart) if chart else lambda: None

    def __call__(self, _x, _y, _size, _color):
        """the symbol interface"""
        chart = self.chart()
        xA = getattr(chart, self.axes[0])
        _xScale = getattr(xA, 'midScale', None)
        if not _xScale:
            _xScale = getattr(xA, 'scale')
        xScale = lambda x: _xScale(x) if x is not None else None
        yA = getattr(chart, self.axes[1])
        _yScale = getattr(yA, 'midScale', None)
        if not _yScale:
            _yScale = getattr(yA, 'scale')
        yScale = lambda x: _yScale(x) if x is not None else None
        G = shapes.Group().add
        strokeWidth = self.strokeWidth
        strokeColor = self.strokeColor
        strokeDashArray = self.strokeDashArray
        crossWidth = self.crossWidth
        crossLo = yScale(self.crossLo)
        crossHi = yScale(self.crossHi)
        boxWidth = self.boxWidth
        boxFillColor = self.boxFillColor
        boxStrokeColor = NotSetOr.conditionalValue(self.boxStrokeColor, strokeColor)
        boxStrokeWidth = NotSetOr.conditionalValue(self.boxStrokeWidth, strokeWidth)
        boxStrokeDashArray = NotSetOr.conditionalValue(self.boxStrokeDashArray, strokeDashArray)
        boxLo = yScale(self.boxLo)
        boxMid = yScale(self.boxMid)
        boxHi = yScale(self.boxHi)
        position = xScale(self.position)
        candleKind = self.candleKind
        haveBox = None not in (boxWidth, boxLo, boxHi)
        haveLine = None not in (crossLo, crossHi)

        def aLine(x0, y0, x1, y1):
            if candleKind != 'vertical':
                x0, y0 = (y0, x0)
                x1, y1 = (y1, x1)
            G(shapes.Line(x0, y0, x1, y1, strokeWidth=strokeWidth, strokeColor=strokeColor, strokeDashArray=strokeDashArray))
        if haveBox:
            boxLo, boxHi = (min(boxLo, boxHi), max(boxLo, boxHi))
        if haveLine:
            crossLo, crossHi = (min(crossLo, crossHi), max(crossLo, crossHi))
            if not haveBox or crossLo >= boxHi or crossHi <= boxLo:
                aLine(position, crossLo, position, crossHi)
                if crossWidth is not None:
                    aLine(position - crossWidth * 0.5, crossLo, position + crossWidth * 0.5, crossLo)
                    aLine(position - crossWidth * 0.5, crossHi, position + crossWidth * 0.5, crossHi)
            elif haveBox:
                if crossLo < boxLo:
                    aLine(position, crossLo, position, boxLo)
                    aLine(position - crossWidth * 0.5, crossLo, position + crossWidth * 0.5, crossLo)
                if crossHi > boxHi:
                    aLine(position, boxHi, position, crossHi)
                    aLine(position - crossWidth * 0.5, crossHi, position + crossWidth * 0.5, crossHi)
        if haveBox:
            x = position - boxWidth * 0.5
            y = boxLo
            h = boxHi - boxLo
            w = boxWidth
            if candleKind != 'vertical':
                x, y, w, h = (y, x, h, w)
            G(shapes.Rect(x, y, w, h, strokeColor=boxStrokeColor if self.boxSides else None, strokeWidth=boxStrokeWidth, strokeDashArray=boxStrokeDashArray, fillColor=boxFillColor))
            if not self.boxSides:
                aLine(position - 0.5 * boxWidth, boxHi, position + 0.5 * boxWidth, boxHi)
                aLine(position - 0.5 * boxWidth, boxLo, position + 0.5 * boxWidth, boxLo)
            if boxMid is not None:
                aLine(position - 0.5 * boxWidth, boxMid, position + 0.5 * boxWidth, boxMid)
        return G.__self__