from reportlab.graphics.shapes import Group, String, Rect
from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.validators import isNumber, isColor, isString, Validator, isBoolean, NoneOr
from reportlab.lib.attrmap import *
from reportlab.graphics.charts.areas import PlotArea
from reportlab.lib.units import mm
from reportlab.lib.utils import asNative
class Ean5BarcodeWidget(Ean13BarcodeWidget):
    """
    EAN-5 barcodes can print the human readable price, set:
        price=True
    """
    codeName = 'EAN5'
    _attrMap = AttrMap(BASE=Ean13BarcodeWidget, price=AttrMapValue(isBoolean, desc='whether to display the price or not'), value=AttrMapValue(nDigits(5), desc='the number'))
    _nbars = 48
    _digits = 5
    _sep = '01'
    _tail = '01011'
    _0csw = 3
    _1csw = 9
    _lhconvert = {'0': (1, 1, 0, 0, 0), '1': (1, 0, 1, 0, 0), '2': (1, 0, 0, 1, 0), '3': (1, 0, 0, 0, 1), '4': (0, 1, 1, 0, 0), '5': (0, 0, 1, 1, 0), '6': (0, 0, 0, 1, 1), '7': (0, 1, 0, 1, 0), '8': (0, 1, 0, 0, 1), '9': (0, 0, 1, 0, 1)}

    def _checkdigit(cls, num):
        z = ord('0')
        iSum = cls._0csw * sum([ord(x) - z for x in num[::2]]) + cls._1csw * sum([ord(x) - z for x in num[1::2]])
        return chr(z + iSum % 10)

    def _encode_left(self, s, a):
        check = self._checkdigit(s)
        cp = self._lhconvert[check]
        _left = self._left
        _sep = self._sep
        z = ord('0')
        full_code = []
        for i, c in enumerate(s):
            full_code.append(_left[cp[i]][ord(c) - z])
        a(_sep.join(full_code))

    def _short_bar(self, i):
        i += 9 - self._lquiet
        return self.humanReadable and (12 < i < 41 or 43 < i < 73)

    def _add_human_readable(self, s, gAdd):
        barWidth = self.barWidth
        fontSize = self.fontSize
        textColor = self.textColor
        fontName = self.fontName
        fth = fontSize * 1.2
        y = self.y + 0.2 * fth
        x = self.x + (self._nbars + self._lquiet * 2) * barWidth / 2
        gAdd(String(x, y, s, fontName=fontName, fontSize=fontSize, fillColor=textColor, textAnchor='middle'))
        price = getattr(self, 'price', None)
        if price:
            price = None
            if s[0] in '3456':
                price = '$'
            elif s[0] in '01':
                price = asNative(b'\xc2\xa3')
            if price is None:
                return
            price += s[1:3] + '.' + s[3:5]
            y += self.barHeight
            gAdd(String(x, y, price, fontName=fontName, fontSize=fontSize, fillColor=textColor, textAnchor='middle'))

    def draw(self):
        g = Group()
        gAdd = g.add
        barWidth = self.barWidth
        width = self.width
        barHeight = self.barHeight
        x = self.x
        y = self.y
        gAdd(Rect(x, y, width, barHeight, fillColor=None, strokeColor=None, strokeWidth=0))
        s = self.value
        self._lquiet = lquiet = self._calc_quiet(self.lquiet)
        rquiet = self._calc_quiet(self.rquiet)
        b = [lquiet * '0' + self._tail]
        a = b.append
        self._encode_left(s, a)
        a(rquiet * '0')
        fontSize = self.fontSize
        barFillColor = self.barFillColor
        barStrokeWidth = self.barStrokeWidth
        barStrokeColor = self.barStrokeColor
        fth = fontSize * 1.2
        b = ''.join(b)
        lrect = None
        for i, c in enumerate(b):
            if c == '1':
                dh = fth
                yh = y + dh
                if lrect and lrect.y == yh:
                    lrect.width += barWidth
                else:
                    lrect = Rect(x, yh, barWidth, barHeight - dh, fillColor=barFillColor, strokeWidth=barStrokeWidth, strokeColor=barStrokeColor)
                    gAdd(lrect)
            else:
                lrect = None
            x += barWidth
        if self.humanReadable:
            self._add_human_readable(s, gAdd)
        return g