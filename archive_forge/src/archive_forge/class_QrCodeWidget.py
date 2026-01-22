import itertools
from reportlab.platypus.flowables import Flowable
from reportlab.graphics.shapes import Group, Rect
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColor, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.units import mm
from reportlab.lib.utils import asUnicodeEx, isUnicode
from reportlab.graphics.barcode import qrencoder
class QrCodeWidget(Widget):
    codeName = 'QR'
    _attrMap = AttrMap(BASE=Widget, value=AttrMapValue(isUnicodeOrQRList, desc='QRCode data'), x=AttrMapValue(isNumber, desc='x-coord'), y=AttrMapValue(isNumber, desc='y-coord'), barFillColor=AttrMapValue(isColor, desc='bar color'), barWidth=AttrMapValue(isNumber, desc='Width of bars.'), barHeight=AttrMapValue(isNumber, desc='Height of bars.'), barBorder=AttrMapValue(isNumber, desc='Width of QR border.'), barLevel=AttrMapValue(isLevel, desc='QR Code level.'), qrVersion=AttrMapValue(isNumberOrNone, desc='QR Code version. None for auto'), barStrokeWidth=AttrMapValue(isNumber, desc='Width of bar borders.'), barStrokeColor=AttrMapValue(isColor, desc='Color of bar borders.'))
    x = 0
    y = 0
    barFillColor = colors.black
    barStrokeColor = None
    barStrokeWidth = 0
    barHeight = 32 * mm
    barWidth = 32 * mm
    barBorder = 4
    barLevel = 'L'
    qrVersion = None
    value = None

    def __init__(self, value='Hello World', **kw):
        self.value = isUnicodeOrQRList.normalize(value)
        for k, v in kw.items():
            setattr(self, k, v)
        ec_level = getattr(qrencoder.QRErrorCorrectLevel, self.barLevel)
        self.__dict__['qr'] = qrencoder.QRCode(self.qrVersion, ec_level)
        if isUnicode(self.value):
            self.addData(self.value)
        elif self.value:
            for v in self.value:
                self.addData(v)

    def addData(self, value):
        self.qr.addData(value)

    def draw(self):
        self.qr.make()
        g = Group()
        color = self.barFillColor
        border = self.barBorder
        width = self.barWidth
        height = self.barHeight
        x = self.x
        y = self.y
        g.add(SRect(x, y, width, height, fillColor=None))
        moduleCount = self.qr.getModuleCount()
        minwh = float(min(width, height))
        boxsize = minwh / (moduleCount + border * 2.0)
        offsetX = x + (width - minwh) / 2.0
        offsetY = y + (minwh - height) / 2.0
        for r, row in enumerate(self.qr.modules):
            row = map(bool, row)
            c = 0
            for t, tt in itertools.groupby(row):
                isDark = t
                count = len(list(tt))
                if isDark:
                    x = (c + border) * boxsize
                    y = (r + border + 1) * boxsize
                    s = SRect(offsetX + x, offsetY + height - y, count * boxsize, boxsize, fillColor=color)
                    g.add(s)
                c += count
        return g