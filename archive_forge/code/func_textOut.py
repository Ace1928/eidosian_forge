from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def textOut(self, text):
    """prints string at current point, text cursor moves across."""
    self._x = self._x + self._canvas.stringWidth(text, self._fontname, self._fontsize)
    self._code.append(self._formatText(text))