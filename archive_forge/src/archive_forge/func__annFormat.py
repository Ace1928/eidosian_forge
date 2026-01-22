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
def _annFormat(D, color, thickness, dashArray, hradius=0, vradius=0):
    from reportlab.pdfbase.pdfdoc import PDFArray
    if color and 'C' not in D:
        D['C'] = PDFArray([color.red, color.green, color.blue])
    if 'Border' not in D:
        border = [hradius, vradius, thickness or 0]
        if dashArray:
            border.append(PDFArray(dashArray))
        D['Border'] = PDFArray(border)