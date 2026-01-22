import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
def _makeBars(self, g, lg):
    bars = self.bars
    br = getattr(self, 'barRecord', None)
    BP = self._barPositions
    flipXY = self._flipXY
    catNAL = self.categoryNALabel
    catNNA = {}
    if catNAL:
        CBL = []
        rowNoL = len(self.data) - 1
        for rowNo, row in enumerate(BP):
            for colNo, (x, y, width, height) in enumerate(row):
                if None not in (width, height):
                    catNNA[colNo] = 1
    lines = [].append
    lineSyms = [].append
    for rowNo, row in enumerate(BP):
        styleCount = len(bars)
        styleIdx = rowNo % styleCount
        rowStyle = bars[styleIdx]
        isLine = bars.checkAttr(rowNo, 'isLine', False)
        linePts = [].append
        for colNo, (x, y, width, height) in enumerate(row):
            style = (styleIdx, colNo) in bars and bars[styleIdx, colNo] or rowStyle
            if None in (width, height):
                if not catNAL or colNo in catNNA:
                    self._addNABarLabel(lg, rowNo, colNo, x, y, width, height)
                elif catNAL and colNo not in CBL:
                    r0 = self._addNABarLabel(lg, rowNo, colNo, x, y, width, height, True, catNAL)
                    if r0:
                        x, y, width, height = BP[rowNoL][colNo]
                        r1 = self._addNABarLabel(lg, rowNoL, colNo, x, y, width, height, True, catNAL)
                        x = (r0[0] + r1[0]) / 2.0
                        y = (r0[1] + r1[1]) / 2.0
                        self._addNABarLabel(lg, rowNoL, colNo, x, y, 0.0001, 0.0001, na=catNAL)
                    CBL.append(colNo)
                if isLine:
                    linePts(None)
                continue
            symbol = None
            if hasattr(style, 'symbol'):
                symbol = copy.deepcopy(style.symbol)
            elif hasattr(self.bars, 'symbol'):
                symbol = self.bars.symbol
            minDimen = getattr(style, 'minDimen', None)
            if minDimen:
                if flipXY:
                    if width < 0:
                        width = min(-style.minDimen, width)
                    else:
                        width = max(style.minDimen, width)
                elif height < 0:
                    height = min(-style.minDimen, height)
                else:
                    height = max(style.minDimen, height)
            if isLine:
                if not flipXY:
                    yL = y + height
                    xL = x + width * 0.5
                else:
                    xL = x + width
                    yL = y + height * 0.5
                linePts(xL)
                linePts(yL)
                if symbol:
                    sym = uSymbol2Symbol(tpcGetItem(symbol, colNo), xL, yL, style.strokeColor or style.fillColor)
                    if sym:
                        lineSyms(sym)
            elif symbol:
                symbol.x = x
                symbol.y = y
                symbol.width = width
                symbol.height = height
                g.add(symbol)
            elif abs(width) > 1e-07 and abs(height) >= 1e-07 and (style.fillColor is not None or style.strokeColor is not None):
                self._makeBar(g, x, y, width, height, rowNo, style)
                if br:
                    br(g.contents[-1], label=self._getLabelText(rowNo, colNo), value=self.data[rowNo][colNo], rowNo=rowNo, colNo=colNo)
            self._addBarLabel(lg, rowNo, colNo, x, y, width, height)
        for linePts in yieldNoneSplits(linePts.__self__):
            if linePts:
                lines(PolyLine(linePts, strokeColor=rowStyle.strokeColor or rowStyle.fillColor, strokeWidth=rowStyle.strokeWidth, strokeDashArray=rowStyle.strokeDashArray))
    for pl in lines.__self__:
        g.add(pl)
    for sym in lineSyms.__self__:
        g.add(sym)