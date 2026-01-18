from reportlab.pdfbase.pdfdoc import (PDFObject, PDFArray, PDFDictionary, PDFString, pdfdocEnc,
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import Color, CMYKColor, Whiter, Blacker, opaqueColor
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import isStr, asNative
import weakref
def zdMark(self, c, size, ds, iFontName):
    c = ZDSyms[c]
    W = H = size - ds
    fs = H / 1.2
    w = float(stringWidth(c, 'ZapfDingbats', fs))
    if w > W:
        fs *= W / w
    dx = ds + 0.5 * (W - w)
    dy = 0
    return 'BT %(iFontName)s %(fs)s Tf %(dx)s %(dy)s Td %(fs)s TL (%(c)s) Tj ET' % vars()