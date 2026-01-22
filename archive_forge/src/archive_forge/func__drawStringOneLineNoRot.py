import math
from io import StringIO
from rdkit.sping.pid import *
from . import psmetrics  # for font info
def _drawStringOneLineNoRot(self, s, x, y, font=None, **kwargs):
    text = self._escape(s)
    self.code.append('%s %s neg moveto (%s) show' % (x, y, text))
    if self._currentFont.underline:
        swidth = self.stringWidth(s, self._currentFont)
        ypos = 0.5 * self.fontDescent(self._currentFont)
        thickness = 0.08 * self._currentFont.size
        self.code.extend(['%s setlinewidth' % thickness, '0 %s neg rmoveto' % ypos, '%s 0 rlineto stroke' % -swidth])