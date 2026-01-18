from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def textLine(self, text=''):
    """prints string at current point, text cursor moves down.
        Can work with no argument to simply move the cursor down."""
    self._x = self._x0
    if self._canvas.bottomup:
        self._y = self._y - self._leading
    else:
        self._y = self._y + self._leading
    self._y0 = self._y
    self._code.append('%s T*' % self._formatText(text))