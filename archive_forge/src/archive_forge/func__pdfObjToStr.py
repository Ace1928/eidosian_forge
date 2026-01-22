from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def _pdfObjToStr(obj):
    if isinstance(obj, PDFArray):
        return '[%s]' % ''.join((_pdfObjToStr(e) for e in obj.sequence))
    if isinstance(obj, PDFFromString):
        return obj._s
    return str(obj)