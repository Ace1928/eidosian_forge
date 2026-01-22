from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def _setFont(self, psfontname, size):
    """Sets the font and fontSize
        Raises a readable exception if an illegal font
        is supplied.  Font names are case-sensitive! Keeps track
        of font anme and size for metrics."""
    self._fontname = psfontname
    self._fontsize = size
    font = pdfmetrics.getFont(self._fontname)
    if font._dynamicFont:
        self._curSubset = -1
    else:
        pdffontname = self._canvas._doc.getInternalFontName(psfontname)
        self._code.append('%s %s Tf' % (pdffontname, fp_str(size)))