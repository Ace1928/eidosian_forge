import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def _make_preamble(self):
    P = [].append
    if self.bottomup:
        P('1 0 0 1 0 0 cm')
    else:
        P('1 0 0 -1 0 %s cm' % fp_str(self._pagesize[1]))
    C = self._code
    n = len(C)
    if self._fillColorObj != (0, 0, 0):
        self.setFillColor(self._fillColorObj)
    if self._strokeColorObj != (0, 0, 0):
        self.setStrokeColor(self._strokeColorObj)
    P(' '.join(C[n:]))
    del C[n:]
    font = pdfmetrics.getFont(self._fontname)
    if not font._dynamicFont:
        if font.face.builtIn or not getattr(self, '_drawTextAsPath', False):
            P('BT %s 12 Tf 14.4 TL ET' % self._doc.getInternalFontName(self._fontname))
    self._preamble = ' '.join(P.__self__)