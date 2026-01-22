from reportlab.graphics.barcode.code39 import Standard39
from reportlab.lib import colors
from reportlab.lib.units import cm
from string import ascii_uppercase, digits as string_digits
class VerticalLTOLabel(BaseLTOLabel):
    """
    A class for LTO labels with rectangular blocks around the tape identifier.
    """
    LABELFONT = ('Helvetica-Bold', 14)
    BLOCKWIDTH = 1 * cm
    BLOCKHEIGHT = 0.45 * cm
    LINEWIDTH = 0.0125
    NBBLOCKS = 7
    COLORSCHEME = ('red', 'yellow', 'lightgreen', 'lightblue', 'grey', 'orangered', 'pink', 'darkgreen', 'orange', 'purple')

    def __init__(self, *args, **kwargs):
        """
        Initializes the label.

        colored : boolean to determine if blocks have to be colorized.
        """
        if 'colored' in kwargs:
            self.colored = kwargs['colored']
            del kwargs['colored']
        else:
            self.colored = False
        kwargs['availheight'] = self.LABELHEIGHT - self.BLOCKHEIGHT
        BaseLTOLabel.__init__(self, *args, **kwargs)

    def drawOn(self, canvas, x, y):
        """Draws some blocks around the identifier's characters."""
        BaseLTOLabel.drawOn(self, canvas, x, y)
        canvas.saveState()
        canvas.setLineWidth(self.LINEWIDTH)
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.translate(x, y)
        xblocks = (self.LABELWIDTH - self.NBBLOCKS * self.BLOCKWIDTH) / 2.0
        for i in range(self.NBBLOCKS):
            font, size = self.LABELFONT
            newfont = self.LABELFONT
            if i == self.NBBLOCKS - 1:
                part = self.label[i:]
                font, size = newfont
                size /= 2.0
                newfont = (font, size)
            else:
                part = self.label[i]
            canvas.saveState()
            canvas.translate(xblocks + i * self.BLOCKWIDTH, 0)
            if self.colored and part.isdigit():
                canvas.setFillColorRGB(*getattr(colors, self.COLORSCHEME[int(part)], colors.Color(1, 1, 1)).rgb())
            else:
                canvas.setFillColorRGB(1, 1, 1)
            canvas.rect(0, 0, self.BLOCKWIDTH, self.BLOCKHEIGHT, fill=True)
            canvas.translate((self.BLOCKWIDTH + canvas.stringWidth(part, *newfont)) / 2.0, self.BLOCKHEIGHT / 2.0)
            canvas.rotate(90.0)
            canvas.setFont(*newfont)
            canvas.setFillColorRGB(0, 0, 0)
            canvas.drawCentredString(0, 0, part)
            canvas.restoreState()
        canvas.restoreState()