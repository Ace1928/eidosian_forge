from reportlab.platypus.flowables import Flowable
from reportlab.lib.units import inch
from string import ascii_lowercase, ascii_uppercase, digits as string_digits
class Barcode(Flowable):
    """Abstract Base for barcodes. Includes implementations of
    some methods suitable for the more primitive barcode types"""
    fontName = 'Courier'
    fontSize = 12
    humanReadable = 0

    def _humanText(self):
        return self.encoded

    def __init__(self, value='', **kwd):
        self.value = str(value)
        self._setKeywords(**kwd)
        if not hasattr(self, 'gap'):
            self.gap = None

    def _calculate(self):
        self.validate()
        self.encode()
        self.decompose()
        self.computeSize()

    def _setKeywords(self, **kwd):
        for k, v in kwd.items():
            setattr(self, k, v)

    def validate(self):
        self.valid = 1
        self.validated = self.value

    def encode(self):
        self.encoded = self.validated

    def decompose(self):
        self.decomposed = self.encoded

    def computeSize(self, *args):
        barWidth = self.barWidth
        wx = barWidth * self.ratio
        if self.gap == None:
            self.gap = barWidth
        w = 0.0
        for c in self.decomposed:
            if c in 'sb':
                w = w + barWidth
            elif c in 'SB':
                w = w + wx
            else:
                w = w + self.gap
        if self.barHeight is None:
            self.barHeight = w * 0.15
            self.barHeight = max(0.25 * inch, self.barHeight)
            if self.bearers:
                self.barHeight = self.barHeight + self.bearers * 2.0 * barWidth
        if self.quiet:
            w += self.lquiet + self.rquiet
        self._height = self.barHeight
        self._width = w

    @property
    def width(self):
        self._calculate()
        return self._width

    @width.setter
    def width(self, v):
        pass

    @property
    def height(self):
        self._calculate()
        return self._height

    @height.setter
    def height(self, v):
        pass

    def draw(self):
        self._calculate()
        barWidth = self.barWidth
        wx = barWidth * self.ratio
        left = self.quiet and self.lquiet or 0
        b = self.bearers * barWidth
        bb = b * 0.5
        tb = self.barHeight - b * 1.5
        for c in self.decomposed:
            if c == 'i':
                left = left + self.gap
            elif c == 's':
                left = left + barWidth
            elif c == 'S':
                left = left + wx
            elif c == 'b':
                self.rect(left, bb, barWidth, tb)
                left = left + barWidth
            elif c == 'B':
                self.rect(left, bb, wx, tb)
                left = left + wx
        if self.bearers:
            if getattr(self, 'bearerBox', None):
                canv = self.canv
                if hasattr(canv, '_Gadd'):
                    canv.rect(bb, bb, self.width, self.barHeight - b, strokeWidth=b, strokeColor=self.barFillColor or self.barStrokeColor, fillColor=None)
                else:
                    canv.saveState()
                    canv.setLineWidth(b)
                    canv.rect(bb, bb, self.width, self.barHeight - b, stroke=1, fill=0)
                    canv.restoreState()
            else:
                w = self._width - (self.lquiet + self.rquiet)
                self.rect(self.lquiet, 0, w, b)
                self.rect(self.lquiet, self.barHeight - b, w, b)
        self.drawHumanReadable()

    def drawHumanReadable(self):
        if self.humanReadable:
            from reportlab.pdfbase.pdfmetrics import getAscent, stringWidth
            s = str(self._humanText())
            fontSize = self.fontSize
            fontName = self.fontName
            w = stringWidth(s, fontName, fontSize)
            width = self._width
            if self.quiet:
                width -= self.lquiet + self.rquiet
                x = self.lquiet
            else:
                x = 0
            if w > width:
                fontSize *= width / float(w)
            y = 1.07 * getAscent(fontName) * fontSize / 1000.0
            self.annotate(x + width / 2.0, -y, s, fontName, fontSize)

    def rect(self, x, y, w, h):
        self.canv.rect(x, y, w, h, stroke=0, fill=1)

    def annotate(self, x, y, text, fontName, fontSize, anchor='middle'):
        canv = self.canv
        canv.saveState()
        canv.setFont(self.fontName, fontSize)
        if anchor == 'middle':
            func = 'drawCentredString'
        elif anchor == 'end':
            func = 'drawRightString'
        else:
            func = 'drawString'
        getattr(canv, func)(x, y, text)
        canv.restoreState()

    def _checkVal(self, name, v, allowed):
        if v not in allowed:
            raise ValueError('%s attribute %s is invalid %r\nnot in allowed %r' % (self.__class__.__name__, name, v, allowed))
        return v