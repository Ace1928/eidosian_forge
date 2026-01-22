import functools
from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfNumbersOrNone,\
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Ellipse, Wedge, String, STATE_DEFAULTS, ArcPath, Polygon, Rect, PolyLine, Line
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab.graphics.charts.textlabels import Label
from reportlab import cmp
from reportlab.graphics.charts.utils3d import _getShaded, _2rad, _360, _180_pi
class LegendedPie(Pie):
    """Pie with a two part legend (one editable with swatches, one hidden without swatches)."""
    _attrMap = AttrMap(BASE=Pie, drawLegend=AttrMapValue(isBoolean, desc='If true then create and draw legend'), legend1=AttrMapValue(None, desc='Handle to legend for pie'), legendNumberFormat=AttrMapValue(None, desc='Formatting routine for number on right hand side of legend.'), legendNumberOffset=AttrMapValue(isNumber, desc='Horizontal space between legend and numbers on r/hand side'), pieAndLegend_colors=AttrMapValue(isListOfColors, desc='Colours used for both swatches and pie'), legend_names=AttrMapValue(isNoneOrListOfNoneOrStrings, desc='Names used in legend (or None)'), legend_data=AttrMapValue(isNoneOrListOfNoneOrNumbers, desc='Numbers used on r/hand side of legend (or None)'), leftPadding=AttrMapValue(isNumber, desc='Padding on left of drawing'), rightPadding=AttrMapValue(isNumber, desc='Padding on right of drawing'), topPadding=AttrMapValue(isNumber, desc='Padding at top of drawing'), bottomPadding=AttrMapValue(isNumber, desc='Padding at bottom of drawing'))

    def __init__(self):
        Pie.__init__(self)
        self.x = 0
        self.y = 0
        self.height = 100
        self.width = 100
        self.data = [38.4, 20.7, 18.9, 15.4, 6.6]
        self.labels = None
        self.direction = 'clockwise'
        PCMYKColor, black = (colors.PCMYKColor, colors.black)
        self.pieAndLegend_colors = [PCMYKColor(11, 11, 72, 0, spotName='PANTONE 458 CV'), PCMYKColor(100, 65, 0, 30, spotName='PANTONE 288 CV'), PCMYKColor(11, 11, 72, 0, spotName='PANTONE 458 CV', density=75), PCMYKColor(100, 65, 0, 30, spotName='PANTONE 288 CV', density=75), PCMYKColor(11, 11, 72, 0, spotName='PANTONE 458 CV', density=50), PCMYKColor(100, 65, 0, 30, spotName='PANTONE 288 CV', density=50)]
        self.slices[0].fillColor = self.pieAndLegend_colors[0]
        self.slices[1].fillColor = self.pieAndLegend_colors[1]
        self.slices[2].fillColor = self.pieAndLegend_colors[2]
        self.slices[3].fillColor = self.pieAndLegend_colors[3]
        self.slices[4].fillColor = self.pieAndLegend_colors[4]
        self.slices[5].fillColor = self.pieAndLegend_colors[5]
        self.slices.strokeWidth = 0.75
        self.slices.strokeColor = black
        legendOffset = 17
        self.legendNumberOffset = 51
        self.legendNumberFormat = '%.1f%%'
        self.legend_data = self.data
        from reportlab.graphics.charts.legends import Legend
        self.legend1 = Legend()
        self.legend1.x = self.width + legendOffset
        self.legend1.y = self.height
        self.legend1.deltax = 5.67
        self.legend1.deltay = 14.17
        self.legend1.dxTextSpace = 11.39
        self.legend1.dx = 5.67
        self.legend1.dy = 5.67
        self.legend1.columnMaximum = 7
        self.legend1.alignment = 'right'
        self.legend_names = ['AAA:', 'AA:', 'A:', 'BBB:', 'NR:']
        for f in range(len(self.data)):
            self.legend1.colorNamePairs.append((self.pieAndLegend_colors[f], self.legend_names[f]))
        self.legend1.fontName = 'Helvetica-Bold'
        self.legend1.fontSize = 6
        self.legend1.strokeColor = black
        self.legend1.strokeWidth = 0.5
        self._legend2 = Legend()
        self._legend2.dxTextSpace = 0
        self._legend2.dx = 0
        self._legend2.alignment = 'right'
        self._legend2.fontName = 'Helvetica-Oblique'
        self._legend2.fontSize = 6
        self._legend2.strokeColor = self.legend1.strokeColor
        self.leftPadding = 5
        self.rightPadding = 5
        self.topPadding = 5
        self.bottomPadding = 5
        self.drawLegend = 1

    def draw(self):
        if self.drawLegend:
            self.legend1.colorNamePairs = []
            self._legend2.colorNamePairs = []
        for f in range(len(self.data)):
            if self.legend_names == None:
                self.slices[f].fillColor = self.pieAndLegend_colors[f]
                self.legend1.colorNamePairs.append((self.pieAndLegend_colors[f], None))
            else:
                try:
                    self.slices[f].fillColor = self.pieAndLegend_colors[f]
                    self.legend1.colorNamePairs.append((self.pieAndLegend_colors[f], self.legend_names[f]))
                except IndexError:
                    self.slices[f].fillColor = self.pieAndLegend_colors[f % len(self.pieAndLegend_colors)]
                    self.legend1.colorNamePairs.append((self.pieAndLegend_colors[f % len(self.pieAndLegend_colors)], self.legend_names[f]))
            if self.legend_data != None:
                ldf = self.legend_data[f]
                lNF = self.legendNumberFormat
                if ldf is None or lNF is None:
                    pass
                elif isinstance(lNF, str):
                    ldf = lNF % ldf
                elif hasattr(lNF, '__call__'):
                    ldf = lNF(ldf)
                else:
                    raise ValueError('Unknown formatter type %s, expected string or function' % ascii(self.legendNumberFormat))
                self._legend2.colorNamePairs.append((None, ldf))
        p = Pie.draw(self)
        if self.drawLegend:
            p.add(self.legend1)
            self._legend2.x = self.legend1.x + self.legendNumberOffset
            self._legend2.y = self.legend1.y
            self._legend2.deltax = self.legend1.deltax
            self._legend2.deltay = self.legend1.deltay
            self._legend2.dy = self.legend1.dy
            self._legend2.columnMaximum = self.legend1.columnMaximum
            p.add(self._legend2)
        p.shift(self.leftPadding, self.bottomPadding)
        return p

    def _getDrawingDimensions(self):
        tx = self.rightPadding
        if self.drawLegend:
            tx += self.legend1.x + self.legendNumberOffset
            tx += self._legend2._calculateMaxWidth(self._legend2.colorNamePairs)
        ty = self.bottomPadding + self.height + self.topPadding
        return (tx, ty)

    def demo(self, drawing=None):
        if not drawing:
            tx, ty = self._getDrawingDimensions()
            drawing = Drawing(tx, ty)
        drawing.add(self.draw())
        return drawing