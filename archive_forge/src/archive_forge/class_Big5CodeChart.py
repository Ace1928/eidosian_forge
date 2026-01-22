import codecs
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable
from reportlab.pdfbase import pdfmetrics, cidfonts
from reportlab.graphics.shapes import Group, String, Rect
from reportlab.graphics.widgetbase import Widget
from reportlab.lib import colors
from reportlab.lib.utils import int2Byte
class Big5CodeChart(CodeChartBase):
    """Formats one 'row' of the 94x160 space used in Big 5

    These deliberately resemble the code charts in Ken Lunde's "Understanding
    CJKV Information Processing", to enable manual checking."""

    def __init__(self, row, faceName, encodingName):
        self.row = row
        self.codePoints = 160
        self.boxSize = 18
        self.charsPerRow = 16
        self.rows = 10
        self.hex = 1
        self.faceName = faceName
        self.encodingName = encodingName
        self.rowLabels = ['4', '5', '6', '7', 'A', 'B', 'C', 'D', 'E', 'F']
        try:
            font = cidfonts.CIDFont(self.faceName, self.encodingName)
            pdfmetrics.registerFont(font)
        except:
            self.faceName = 'Helvetica'
            self.encodingName = 'WinAnsiEncoding'
        self.fontName = self.faceName + '-' + self.encodingName
        self.calcLayout()

    def makeRow(self, row):
        """Works out the character values for this Big5 row.
        Rows start at 0xA1"""
        cells = []
        if self.encodingName.find('B5') > -1:
            for y in [4, 5, 6, 7, 10, 11, 12, 13, 14, 15]:
                for x in range(16):
                    col = y * 16 + x
                    ch = int2Byte(row) + int2Byte(col)
                    cells.append(ch)
        else:
            cells.append([None] * 160)
        return cells

    def draw(self):
        self.drawLabels(topLeft='%02X' % self.row)
        charList = self.makeRow(self.row)
        self.drawChars(charList)
        self.canv.grid(self.xlist, self.ylist)