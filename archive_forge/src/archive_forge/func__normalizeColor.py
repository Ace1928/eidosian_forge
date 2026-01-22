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
def _normalizeColor(aColor):
    if isinstance(aColor, CMYKColor):
        d = aColor.density
        return ('DeviceCMYK', tuple((c * d for c in aColor.cmyk())))
    elif isinstance(aColor, Color):
        return ('DeviceRGB', aColor.rgb())
    elif isinstance(aColor, (tuple, list)):
        l = len(aColor)
        if l == 3:
            return ('DeviceRGB', aColor)
        elif l == 4:
            return ('DeviceCMYK', aColor)
    elif isinstance(aColor, str):
        return _normalizeColor(toColor(aColor))
    raise ValueError('Unknown color %r' % aColor)