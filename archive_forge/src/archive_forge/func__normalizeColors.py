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
def _normalizeColors(colors):
    space = None
    outcolors = []
    for aColor in colors:
        nspace, outcolor = _normalizeColor(aColor)
        if space is not None and space != nspace:
            raise ValueError('Mismatch in color spaces: %s and %s' % (space, nspace))
        space = nspace
        outcolors.append(outcolor)
    return (space, outcolors)