from reportlab.lib.units import inch,cm
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.formatters import DecimalFormatter
from reportlab.graphics.shapes import definePath, Group, Drawing, Rect, PolyLine, String
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.colors import Color, black, white, ReportLabBlue
from reportlab.pdfbase.pdfmetrics import stringWidth
@staticmethod
def applyPrec(P, prec):
    if prec is None:
        return P
    R = [].append
    f = DecimalFormatter(places=prec)
    for p in P:
        if isSeq(p):
            n = [].append
            for e in p:
                if isinstance(e, float):
                    e = float(f(e))
                n(e)
            p = n.__self__
        R(p)
    return R.__self__