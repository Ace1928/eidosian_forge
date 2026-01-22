import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def _drawStringOneLine(self, s, x, y, font=None, color=None, angle=0, **kwargs):
    text = self._escape(s)
    self.code.extend(['%f %f neg moveto (%s) show' % (x, y, text)])
    if self._currentFont.underline:
        swidth = self.stringWidth(s, self._currentFont)
        dy = 0.5 * self.fontDescent(self._currentFont)
        thickness = 0.08 * self._currentFont.size
        self.code.extend(['%s setlinewidth' % thickness, '%f %f neg moveto' % (x, dy + y), '%f 0 rlineto stroke' % swidth])