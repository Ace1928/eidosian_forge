import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
class CodeChartBase(Flowable):
    """Basic bits of drawing furniture used by
    single and multi-byte versions: ability to put letters
    into boxes."""

    def calcLayout(self):
        """Work out x and y positions for drawing"""
        rows = self.codePoints * 1.0 / self.charsPerRow
        if rows == int(rows):
            self.rows = int(rows)
        else:
            self.rows = int(rows) + 1
        self.width = self.boxSize * (1 + self.charsPerRow)
        self.height = self.boxSize * (1 + self.rows)
        self.ylist = []
        for row in range(self.rows + 2):
            self.ylist.append(row * self.boxSize)
        self.xlist = []
        for col in range(self.charsPerRow + 2):
            self.xlist.append(col * self.boxSize)

    def formatByte(self, byt):
        if self.hex:
            return '%02X' % byt
        else:
            return '%d' % byt

    def drawChars(self, charList):
        """Fills boxes in order.  None means skip a box.
        Empty boxes at end get filled with gray"""
        extraNeeded = self.rows * self.charsPerRow - len(charList)
        for i in range(extraNeeded):
            charList.append(None)
        row = 0
        col = 0
        self.canv.setFont(self.fontName, self.boxSize * 0.75)
        for ch in charList:
            if ch is None:
                self.canv.setFillGray(0.9)
                self.canv.rect((1 + col) * self.boxSize, (self.rows - row - 1) * self.boxSize, self.boxSize, self.boxSize, stroke=0, fill=1)
                self.canv.setFillGray(0.0)
            else:
                try:
                    self.canv.drawCentredString((col + 1.5) * self.boxSize, (self.rows - row - 0.875) * self.boxSize, ch)
                except:
                    self.canv.setFillGray(0.9)
                    self.canv.rect((1 + col) * self.boxSize, (self.rows - row - 1) * self.boxSize, self.boxSize, self.boxSize, stroke=0, fill=1)
                    self.canv.drawCentredString((col + 1.5) * self.boxSize, (self.rows - row - 0.875) * self.boxSize, '?')
                    self.canv.setFillGray(0.0)
            col = col + 1
            if col == self.charsPerRow:
                row = row + 1
                col = 0

    def drawLabels(self, topLeft=''):
        """Writes little labels in the top row and first column"""
        self.canv.setFillGray(0.8)
        self.canv.rect(0, self.ylist[-2], self.width, self.boxSize, fill=1, stroke=0)
        self.canv.rect(0, 0, self.boxSize, self.ylist[-2], fill=1, stroke=0)
        self.canv.setFillGray(0.0)
        self.canv.setFont('Helvetica-Oblique', 0.375 * self.boxSize)
        byt = 0
        for row in range(self.rows):
            if self.rowLabels:
                label = self.rowLabels[row]
            else:
                label = self.formatByte(row * self.charsPerRow)
            self.canv.drawCentredString(0.5 * self.boxSize, (self.rows - row - 0.75) * self.boxSize, label)
        for col in range(self.charsPerRow):
            self.canv.drawCentredString((col + 1.5) * self.boxSize, (self.rows + 0.25) * self.boxSize, self.formatByte(col))
        if topLeft:
            self.canv.setFont('Helvetica-BoldOblique', 0.5 * self.boxSize)
            self.canv.drawCentredString(0.5 * self.boxSize, (self.rows + 0.25) * self.boxSize, topLeft)