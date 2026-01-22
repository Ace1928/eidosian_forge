from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
def _add_human_readable(self, s, gAdd):
    Ean13BarcodeWidget._add_human_readable(self, s, gAdd)
    barWidth = self.barWidth
    barHeight = self.barHeight
    fontSize = self.fontSize
    textColor = self.textColor
    fontName = self.fontName
    fth = fontSize * 1.2
    y = self.y + 0.2 * fth + barHeight
    x = self._lquiet * barWidth
    isbn = 'ISBN '
    segments = [s[0:3], s[3:4], s[4:9], s[9:12], s[12]]
    isbn += '-'.join(segments)
    gAdd(String(x, y, isbn, fontName=fontName, fontSize=fontSize, fillColor=textColor))