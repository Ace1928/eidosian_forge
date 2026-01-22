from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Line, Rect, Polygon, PolyLine, \
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.piecharts import WedgeLabel
from reportlab.graphics.widgets.markers import makeMarker, uSymbol2Symbol, isSymbol
class SpiderChart(PlotArea):
    _attrMap = AttrMap(BASE=PlotArea, data=AttrMapValue(None, desc='Data to be plotted, list of (lists of) numbers.'), labels=AttrMapValue(isListOfStringsOrNone, desc='optional list of labels to use for each data point'), startAngle=AttrMapValue(isNumber, desc='angle of first slice; like the compass, 0 is due North'), direction=AttrMapValue(OneOf('clockwise', 'anticlockwise'), desc="'clockwise' or 'anticlockwise'"), strands=AttrMapValue(None, desc='collection of strand descriptor objects'), spokes=AttrMapValue(None, desc='collection of spoke descriptor objects'), strandLabels=AttrMapValue(None, desc='collection of strand label descriptor objects'), strandLabelClass=AttrMapValue(NoneOr(isCallable), desc='A class factory to use for the strand labels'), spokeLabels=AttrMapValue(None, desc='collection of spoke label descriptor objects'), spokeLabelClass=AttrMapValue(NoneOr(isCallable), desc='A class factory to use for the spoke labels'))

    def makeSwatchSample(self, rowNo, x, y, width, height):
        baseStyle = self.strands
        styleIdx = rowNo % len(baseStyle)
        style = baseStyle[styleIdx]
        strokeColor = getattr(style, 'strokeColor', getattr(baseStyle, 'strokeColor', None))
        fillColor = getattr(style, 'fillColor', getattr(baseStyle, 'fillColor', None))
        strokeDashArray = getattr(style, 'strokeDashArray', getattr(baseStyle, 'strokeDashArray', None))
        strokeWidth = getattr(style, 'strokeWidth', getattr(baseStyle, 'strokeWidth', 0))
        symbol = getattr(style, 'symbol', getattr(baseStyle, 'symbol', None))
        ym = y + height / 2.0
        if fillColor is None and strokeColor is not None and (strokeWidth > 0):
            bg = Line(x, ym, x + width, ym, strokeWidth=strokeWidth, strokeColor=strokeColor, strokeDashArray=strokeDashArray)
        elif fillColor is not None:
            bg = Rect(x, y, width, height, strokeWidth=strokeWidth, strokeColor=strokeColor, strokeDashArray=strokeDashArray, fillColor=fillColor)
        else:
            bg = None
        if symbol:
            symbol = uSymbol2Symbol(symbol, x + width / 2.0, ym, color)
            if bg:
                g = Group()
                g.add(bg)
                g.add(symbol)
                return g
        return symbol or bg

    def getSeriesName(self, i, default=None):
        """return series name i or default"""
        return _objStr(getattr(self.strands[i], 'name', default))

    def __init__(self):
        PlotArea.__init__(self)
        self.data = [[10, 12, 14, 16, 14, 12], [6, 8, 10, 12, 9, 11]]
        self.labels = None
        self.labels = ['a', 'b', 'c', 'd', 'e', 'f']
        self.startAngle = 90
        self.direction = 'clockwise'
        self.strands = TypedPropertyCollection(StrandProperty)
        self.spokes = TypedPropertyCollection(SpokeProperty)
        self.spokeLabels = TypedPropertyCollection(SpokeLabel)
        self.spokeLabels._text = None
        self.strandLabels = TypedPropertyCollection(StrandLabel)
        self.x = 10
        self.y = 10
        self.width = 180
        self.height = 180

    def demo(self):
        d = Drawing(200, 200)
        d.add(SpiderChart())
        return d

    def normalizeData(self, outer=0.0):
        """Turns data into normalized ones where each datum is < 1.0,
        and 1.0 = maximum radius.  Adds 10% at outside edge by default"""
        data = self.data
        assert min(list(map(min, data))) >= 0, 'Cannot do spider plots of negative numbers!'
        norm = max(list(map(max, data)))
        norm *= 1.0 + outer
        if norm < 1e-09:
            norm = 1.0
        self._norm = norm
        return [[e / norm for e in row] for row in data]

    def _innerDrawLabel(self, sty, radius, cx, cy, angle, car, sar, labelClass=None):
        """Draw a label for a given item in the list."""
        fmt = sty.format
        value = radius * self._norm
        if not fmt:
            text = None
        elif isinstance(fmt, str):
            if fmt == 'values':
                text = sty._text
            else:
                text = fmt % value
        elif hasattr(fmt, '__call__'):
            text = fmt(value)
        else:
            raise ValueError('Unknown formatter type %s, expected string or function' % fmt)
        if text:
            dR = sty.dR
            if dR:
                radius += dR / self._radius
            L = _setupLabel(labelClass, text, radius, cx, cy, angle, car, sar, sty)
            if dR < 0:
                L._anti = 1
        else:
            L = None
        return L

    def labelClass(self, kind):
        klass = getattr(self, f'{kind}LabelClass', None)
        if not klass:
            klass = globals()[f'{kind.capitalize()}Label']
        return klass

    def draw(self):
        g = self.makeBackground() or Group()
        xradius = self.width / 2.0
        yradius = self.height / 2.0
        self._radius = radius = min(xradius, yradius)
        cx = self.x + xradius
        cy = self.y + yradius
        data = self.normalizeData()
        self._seriesCount = len(data)
        n = len(data[0])
        if self.labels is None:
            labels = [''] * n
        else:
            labels = self.labels
            i = n - len(labels)
            if i > 0:
                labels = labels + [''] * i
        S = []
        STRANDS = []
        STRANDAREAS = []
        syms = []
        labs = []
        csa = []
        angle = self.startAngle * pi / 180
        direction = self.direction == 'clockwise' and -1 or 1
        angleBetween = direction * (2 * pi) / float(n)
        spokes = self.spokes
        spokeLabels = self.spokeLabels
        for i in range(n):
            car = cos(angle) * radius
            sar = sin(angle) * radius
            csa.append((car, sar, angle))
            si = self.spokes[i]
            if si.visible:
                spoke = Line(cx, cy, cx + car, cy + sar, strokeWidth=si.strokeWidth, strokeColor=si.strokeColor, strokeDashArray=si.strokeDashArray)
            S.append(spoke)
            sli = spokeLabels[i]
            text = sli._text
            if not text:
                text = labels[i]
            if text:
                S.append(_setupLabel(self.labelClass('spoke'), text, si.labelRadius, cx, cy, angle, car, sar, sli))
            angle += angleBetween
        rowIdx = 0
        strands = self.strands
        strandLabels = self.strandLabels
        for row in data:
            rsty = strands[rowIdx]
            points = []
            car, sar = csa[-1][:2]
            r = row[-1]
            points.append(cx + car * r)
            points.append(cy + sar * r)
            for i in range(n):
                car, sar, angle = csa[i]
                r = row[i]
                points.append(cx + car * r)
                points.append(cy + sar * r)
                L = self._innerDrawLabel(strandLabels[rowIdx, i], r, cx, cy, angle, car, sar, labelClass=self.labelClass('strand'))
                if L:
                    labs.append(L)
                sty = strands[rowIdx, i]
                uSymbol = sty.symbol
                if uSymbol:
                    s_x = cx + car * r
                    s_y = cy + sar * r
                    s_fillColor = sty.fillColor
                    s_strokeColor = sty.strokeColor
                    s_strokeWidth = sty.strokeWidth
                    s_angle = 0
                    s_size = sty.symbolSize
                    if type(uSymbol) is type(''):
                        symbol = makeMarker(uSymbol, size=s_size, x=s_x, y=s_y, fillColor=s_fillColor, strokeColor=s_strokeColor, strokeWidth=s_strokeWidth, angle=s_angle)
                    else:
                        symbol = uSymbol2Symbol(uSymbol, s_x, s_y, s_fillColor)
                        for k, v in (('size', s_size), ('fillColor', s_fillColor), ('x', s_x), ('y', s_y), ('strokeColor', s_strokeColor), ('strokeWidth', s_strokeWidth), ('angle', s_angle)):
                            if getattr(symbol, k, None) is None:
                                try:
                                    setattr(symbol, k, v)
                                except:
                                    pass
                    syms.append(symbol)
            if rsty.fillColor:
                strand = Polygon(points)
                strand.fillColor = rsty.fillColor
                strand.strokeColor = None
                strand.strokeWidth = 0
                STRANDAREAS.append(strand)
            if rsty.strokeColor and rsty.strokeWidth:
                strand = PolyLine(points)
                strand.strokeColor = rsty.strokeColor
                strand.strokeWidth = rsty.strokeWidth
                strand.strokeDashArray = rsty.strokeDashArray
                STRANDS.append(strand)
            rowIdx += 1
        for s in STRANDAREAS + STRANDS + syms + S + labs:
            g.add(s)
        return g