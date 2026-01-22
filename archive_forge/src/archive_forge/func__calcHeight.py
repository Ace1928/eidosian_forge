import copy
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, OneOf, isString, isColorOrNone,\
from reportlab.lib.attrmap import *
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection, PropHolder
from reportlab.graphics.shapes import Drawing, Group, String, Rect, Line, STATE_DEFAULTS
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.utils import isSeq, find_locals, isStr, asNative
from reportlab.graphics.shapes import _baseGFontName
def _calcHeight(self):
    dy = self.dy
    yGap = self.yGap
    thisy = upperlefty = self.y - dy
    fontSize = self.fontSize
    fontName = self.fontName
    ascent = getFont(fontName).face.ascent / 1000.0
    if ascent == 0:
        ascent = 0.718
    ascent *= fontSize
    leading = fontSize * 1.2
    deltay = self.deltay
    if not deltay:
        deltay = max(dy, leading) + self.autoYPadding
    columnCount = 0
    count = 0
    lowy = upperlefty
    lim = self.columnMaximum - 1
    for name in self._getTexts(self.colorNamePairs):
        y0 = thisy + (dy - ascent) * 0.5
        y = y0 - _getLineCount(name) * leading
        leadingMove = 2 * y0 - y - thisy
        newy = thisy - max(deltay, leadingMove) - yGap
        lowy = min(y, newy, lowy)
        if count == lim:
            count = 0
            thisy = upperlefty
            columnCount += 1
        else:
            thisy = newy
            count = count + 1
    return upperlefty - lowy