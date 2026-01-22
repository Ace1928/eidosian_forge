from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import flatten, isStr
from reportlab.graphics.shapes import Drawing, Group, Rect, PolyLine, Polygon, _SetKeyWordArgs
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.charts.textlabels import Label
from reportlab.graphics.charts.axes import XValueAxis, YValueAxis, AdjYValueAxis, NormalDateXValueAxis
from reportlab.graphics.charts.utils import *
from reportlab.graphics.widgets.markers import uSymbol2Symbol, makeMarker
from reportlab.graphics.widgets.grids import Grid, DoubleGrid, ShadedPolygon
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.charts.areas import PlotArea
from .utils import FillPairedData
from reportlab.graphics.charts.linecharts import AbstractLineChart
class LinePlot(AbstractLineChart):
    """Line plot with multiple lines.

    Both x- and y-axis are value axis (so there are no seperate
    X and Y versions of this class).
    """
    _attrMap = AttrMap(BASE=PlotArea, reversePlotOrder=AttrMapValue(isBoolean, desc='If true reverse plot order.', advancedUsage=1), lineLabelNudge=AttrMapValue(isNumber, desc='Distance between a data point and its label.', advancedUsage=1), lineLabels=AttrMapValue(None, desc='Handle to the list of data point labels.'), lineLabelFormat=AttrMapValue(None, desc='Formatting string or function used for data point labels.'), lineLabelArray=AttrMapValue(None, desc='explicit array of line label values, must match size of data if present.'), strokeColor=AttrMapValue(isColorOrNone, desc='Color used for background border of plot area.'), fillColor=AttrMapValue(isColorOrNone, desc='Color used for background interior of plot area.'), lines=AttrMapValue(None, desc='Handle of the lines.'), xValueAxis=AttrMapValue(None, desc='Handle of the x axis.'), yValueAxis=AttrMapValue(None, desc='Handle of the y axis.'), data=AttrMapValue(None, desc='Data to be plotted, list of (lists of) x/y tuples.'), annotations=AttrMapValue(None, desc='list of callables, will be called with self, xscale, yscale.', advancedUsage=1), behindAxes=AttrMapValue(isBoolean, desc='If true use separate line group.', advancedUsage=1), gridFirst=AttrMapValue(isBoolean, desc='If true use draw grids before axes.', advancedUsage=1))

    def __init__(self):
        PlotArea.__init__(self)
        self.reversePlotOrder = 0
        self.xValueAxis = XValueAxis()
        self.yValueAxis = YValueAxis()
        self.data = [((1, 1), (2, 2), (2.5, 1), (3, 3), (4, 5)), ((1, 2), (2, 3), (2.5, 2), (3, 4), (4, 6))]
        self.lines = TypedPropertyCollection(LinePlotProperties)
        self.lines.strokeWidth = 1
        self.joinedLines = 1
        self.lines[0].strokeColor = colors.red
        self.lines[1].strokeColor = colors.blue
        self.lineLabels = TypedPropertyCollection(Label)
        self.lineLabelFormat = None
        self.lineLabelArray = None
        self.lineLabelNudge = 10
        self._inFill = None
        self.annotations = []
        self.behindAxes = 0
        self.gridFirst = 0

    @property
    def joinedLines(self):
        return self.lines.lineStyle == 'joinedLine'

    @joinedLines.setter
    def joinedLines(self, v):
        self.lines.lineStyle = 'joinedLine' if v else 'line'

    def demo(self):
        """Shows basic use of a line chart."""
        drawing = Drawing(400, 200)
        data = [((1, 1), (2, 2), (2.5, 1), (3, 3), (4, 5)), ((1, 2), (2, 3), (2.5, 2), (3.5, 5), (4, 6))]
        lp = LinePlot()
        lp.x = 50
        lp.y = 50
        lp.height = 125
        lp.width = 300
        lp.data = data
        lp.joinedLines = 1
        lp.lineLabelFormat = '%2.0f'
        lp.strokeColor = colors.black
        lp.lines[0].strokeColor = colors.red
        lp.lines[0].symbol = makeMarker('FilledCircle')
        lp.lines[1].strokeColor = colors.blue
        lp.lines[1].symbol = makeMarker('FilledDiamond')
        lp.xValueAxis.valueMin = 0
        lp.xValueAxis.valueMax = 5
        lp.xValueAxis.valueStep = 1
        lp.yValueAxis.valueMin = 0
        lp.yValueAxis.valueMax = 7
        lp.yValueAxis.valueStep = 1
        drawing.add(lp)
        return drawing

    def calcPositions(self):
        """Works out where they go.

        Sets an attribute _positions which is a list of
        lists of (x, y) matching the data.
        """
        self._seriesCount = len(self.data)
        self._rowLength = max(list(map(len, self.data)))
        pairs = set()
        P = [].append
        xscale = self.xValueAxis.scale
        yscale = self.yValueAxis.scale
        data = self.data
        n = len(data)
        for rowNo, row in enumerate(data):
            if isinstance(row, FillPairedData):
                other = row.other
                if 0 <= other < n:
                    if other == rowNo:
                        raise ValueError('data row %r may not be paired with itself' % rowNo)
                    pairs.add((rowNo, other))
                else:
                    raise ValueError('data row %r is paired with invalid data row %r' % (rowNo, other))
            line = [].append
            for colNo, datum in enumerate(row):
                xv = datum[0]
                line((xscale(mktime(mkTimeTuple(xv))) if isStr(xv) else xscale(xv), yscale(datum[1])))
            P(line.__self__)
        P = P.__self__
        for rowNo, other in pairs:
            P[rowNo] = FillPairedData(P[rowNo], other)
        self._pairInFills = len(pairs)
        self._positions = P

    def _innerDrawLabel(self, rowNo, colNo, x, y):
        """Draw a label for a given item in the list."""
        labelFmt = self.lineLabelFormat
        labelValue = self.data[rowNo][colNo][1]
        if labelFmt is None:
            labelText = None
        elif isinstance(labelFmt, str):
            if labelFmt == 'values':
                labelText = self.lineLabelArray[rowNo][colNo]
            else:
                labelText = labelFmt % labelValue
        elif hasattr(labelFmt, '__call__'):
            if not hasattr(labelFmt, '__labelFmtEX__'):
                labelText = labelFmt(labelValue)
            else:
                labelText = labelFmt(self, rowNo, colNo, x, y)
        else:
            raise ValueError('Unknown formatter type %s, expected string or function' % labelFmt)
        if labelText:
            label = self.lineLabels[rowNo, colNo]
            if not label.visible:
                return
            if y > 0:
                label.setOrigin(x, y + self.lineLabelNudge)
            else:
                label.setOrigin(x, y - self.lineLabelNudge)
            label.setText(labelText)
        else:
            label = None
        return label

    def drawLabel(self, G, rowNo, colNo, x, y):
        """Draw a label for a given item in the list.
        G must have an add method"""
        G.add(self._innerDrawLabel(rowNo, colNo, x, y))

    def makeLines(self):
        g = Group()
        yA = self.yValueAxis
        xA = self.xValueAxis
        bubblePlot = getattr(self, '_bubblePlot', None)
        if bubblePlot:
            bubbleR = min(yA._bubbleRadius, xA._bubbleRadius)
            bubbleMax = xA._bubbleMax
        labelFmt = self.lineLabelFormat
        P = self._positions
        _inFill = getattr(self, '_inFill', None)
        lines = self.lines
        styleCount = len(lines)
        if _inFill or self._pairInFills or [rowNo for rowNo in range(len(P)) if getattr(lines[rowNo % styleCount], 'inFill', False)]:
            inFillY = getattr(_inFill, 'yValue', None)
            if inFillY is None:
                inFillY = xA._y
            else:
                inFillY = yA.scale(inFillY)
            inFillX0 = yA._x
            inFillX1 = inFillX0 + xA._length
            inFillG = getattr(self, '_inFillG', g)
        bw = None
        lG = getattr(self, '_lineG', g)
        R = range(len(P))
        if self.reversePlotOrder:
            R = reversed(R)
        for rowNo in R:
            row = P[rowNo]
            styleRowNo = rowNo % styleCount
            rowStyle = lines[styleRowNo]
            strokeColor = getattr(rowStyle, 'strokeColor', None)
            strokeWidth = getattr(rowStyle, 'strokeWidth', None)
            fillColor = getattr(rowStyle, 'fillColor', strokeColor)
            inFill = getattr(rowStyle, 'inFill', _inFill)
            dash = getattr(rowStyle, 'strokeDashArray', None)
            lineStyle = getattr(rowStyle, 'lineStyle', None)
            if hasattr(rowStyle, 'strokeWidth'):
                width = rowStyle.strokeWidth
            elif hasattr(lines, 'strokeWidth'):
                width = lines.strokeWidth
            else:
                width = None
            if lineStyle == 'bar':
                if bw is None:
                    x = max(map(len, P)) - 1
                    bw = self.width / x - 1 if x > 0 else self.width
                    barWidth = getattr(rowStyle, 'barWidth', Percentage(50))
                    x = self.yValueAxis
                    y0 = x.scale(0)
                    bypos = max(x._y, y0)
                    byneg = min(x._y + x._length, y0)
                    xmin = self.xValueAxis._x
                    xmax = xmin + self.xValueAxis._length
                    if isinstance(barWidth, Percentage):
                        bw *= barWidth * 0.005
                    else:
                        bw = barWidth * 0.5
                for x, y in row:
                    w = bw
                    _y0 = byneg if y < y0 else bypos
                    x -= w / 2
                    if x < xmin:
                        w -= xmin - x
                        x = xmin
                    elif x + w > xmax:
                        w -= xmax - x
                    g.add(Rect(x, _y0, w, y - _y0, strokeWidth=strokeWidth, strokeColor=strokeColor, fillColor=fillColor))
            elif lineStyle == 'joinedLine':
                points = flatten(row)
                if inFill or isinstance(row, FillPairedData):
                    filler = getattr(rowStyle, 'filler', None)
                    if isinstance(row, FillPairedData):
                        fpoints = points + flatten(reversed(P[row.other]))
                    else:
                        fpoints = [inFillX0, inFillY] + points + [inFillX1, inFillY]
                    if filler:
                        filler.fill(self, inFillG, rowNo, fillColor, fpoints)
                    else:
                        inFillG.add(Polygon(fpoints, fillColor=fillColor, strokeColor=strokeColor if strokeColor == fillColor else None, strokeWidth=width or 0.1))
                if not inFill or inFill == 2 or strokeColor != fillColor:
                    line = PolyLine(points, strokeColor=strokeColor, strokeLineCap=0, strokeLineJoin=1)
                    if width:
                        line.strokeWidth = width
                    if dash:
                        line.strokeDashArray = dash
                    lG.add(line)
            if hasattr(rowStyle, 'symbol'):
                uSymbol = rowStyle.symbol
            elif hasattr(lines, 'symbol'):
                uSymbol = lines.symbol
            else:
                uSymbol = None
            if uSymbol:
                if bubblePlot:
                    drow = self.data[rowNo]
                for j, xy in enumerate(row):
                    if (styleRowNo, j) in lines:
                        juSymbol = getattr(lines[styleRowNo, j], 'symbol', uSymbol)
                    else:
                        juSymbol = uSymbol
                    if juSymbol is uSymbol:
                        symbol = uSymbol
                        symColor = strokeColor
                    else:
                        symbol = juSymbol
                        symColor = getattr(symbol, 'fillColor', strokeColor)
                    symbol = uSymbol2Symbol(tpcGetItem(symbol, j), xy[0], xy[1], symColor)
                    if symbol:
                        if bubblePlot:
                            symbol.size = bubbleR * (drow[j][2] / bubbleMax) ** 0.5
                        g.add(symbol)
            else:
                if bubblePlot:
                    drow = self.data[rowNo]
                for j, xy in enumerate(row):
                    juSymbol = getattr(lines[styleRowNo, j], 'symbol', None)
                    if not juSymbol:
                        continue
                    symColor = getattr(juSymbol, 'fillColor', getattr(juSymbol, 'strokeColor', strokeColor))
                    symbol = uSymbol2Symbol(juSymbol, xy[0], xy[1], symColor)
                    if symbol:
                        if bubblePlot:
                            symbol.size = bubbleR * (drow[j][2] / bubbleMax) ** 0.5
                        g.add(symbol)
            for colNo, datum in enumerate(row):
                x1, y1 = datum
                self.drawLabel(g, rowNo, colNo, x1, y1)
            shader = getattr(rowStyle, 'shader', None)
            if shader:
                shader.shade(self, g, rowNo, strokeColor, row)
        return g

    def draw(self):
        yA = self.yValueAxis
        xA = self.xValueAxis
        if getattr(self, '_bubblePlot', None):
            yA._bubblePlot = xA._bubblePlot = 1
        yA.setPosition(self.x, self.y, self.height)
        if yA:
            yA.joinAxis = xA
        if xA:
            xA.joinAxis = yA
        yA.configure(self.data)
        xAxisCrossesAt = yA.scale(0)
        if xAxisCrossesAt > self.y + self.height or xAxisCrossesAt < self.y:
            y = self.y
        else:
            y = xAxisCrossesAt
        xA.setPosition(self.x, y, self.width)
        xA.configure(self.data)
        self.calcPositions()
        g = Group()
        g.add(self.makeBackground())
        if self._inFill or self.behindAxes:
            xA._joinToAxis()
            if self._inFill:
                self._inFillG = Group()
                g.add(self._inFillG)
            if self.behindAxes:
                self._lineG = Group()
                g.add(self._lineG)
        xA._joinToAxis()
        yA._joinToAxis()
        xAex = xA.visibleAxis and [xA._y] or []
        yAex = yA.visibleAxis and [yA._x] or []
        skipGrid = getattr(xA, 'skipGrid', 'none')
        if skipGrid != None:
            if skipGrid in ('both', 'top'):
                yAex.append(xA._x + xA._length)
            if skipGrid in ('both', 'bottom'):
                yAex.append(xA._x)
        skipGrid = getattr(yA, 'skipGrid', 'none')
        if skipGrid != None:
            if skipGrid in ('both', 'top'):
                xAex.append(yA._y + yA._length)
            if skipGrid in ('both', 'bottom'):
                xAex.append(yA._y)
        if self.gridFirst:
            xA.makeGrid(g, parent=self, dim=yA.getGridDims, exclude=yAex)
            yA.makeGrid(g, parent=self, dim=xA.getGridDims, exclude=xAex)
        g.add(xA.draw())
        g.add(yA.draw())
        if not self.gridFirst:
            xAdgl = getattr(xA, 'drawGridLast', False)
            yAdgl = getattr(yA, 'drawGridLast', False)
            if not xAdgl:
                xA.makeGrid(g, parent=self, dim=yA.getGridDims, exclude=yAex)
            if not yAdgl:
                yA.makeGrid(g, parent=self, dim=xA.getGridDims, exclude=xAex)
        annotations = getattr(self, 'annotations', [])
        for a in annotations:
            if getattr(a, 'beforeLines', None):
                g.add(a(self, xA.scale, yA.scale))
        g.add(self.makeLines())
        if not self.gridFirst:
            if xAdgl:
                xA.makeGrid(g, parent=self, dim=yA.getGridDims, exclude=yAex)
            if yAdgl:
                yA.makeGrid(g, parent=self, dim=xA.getGridDims, exclude=xAex)
        for a in annotations:
            if not getattr(a, 'beforeLines', None):
                g.add(a(self, xA.scale, yA.scale))
        return g

    def addCrossHair(self, name, xv, yv, strokeColor=colors.black, strokeWidth=1, beforeLines=True):
        from reportlab.graphics.shapes import Group, Line
        annotations = [a for a in getattr(self, 'annotations', []) if getattr(a, 'name', None) != name]

        def annotation(self, xScale, yScale):
            x = xScale(xv)
            y = yScale(yv)
            g = Group()
            xA = xScale.__self__
            g.add(Line(xA._x, y, xA._x + xA._length, y, strokeColor=strokeColor, strokeWidth=strokeWidth))
            yA = yScale.__self__
            g.add(Line(x, yA._y, x, yA._y + yA._length, strokeColor=strokeColor, strokeWidth=strokeWidth))
            return g
        annotation.beforeLines = beforeLines
        annotations.append(annotation)
        self.annotations = annotations