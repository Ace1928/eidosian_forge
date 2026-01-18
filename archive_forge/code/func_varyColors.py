from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
@staticmethod
def varyColors(key, t, b, f):
    if key != 'N':
        func = Whiter if key == 'R' else Blacker
        t, b, f = [func(c, 0.9) for c in (t, b, f)]
    return (t, b, f)