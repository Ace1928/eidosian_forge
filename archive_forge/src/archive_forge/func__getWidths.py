import copy
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, OneOf, isString, isColorOrNone,\
from reportlab.lib.attrmap import *
from reportlab.pdfbase.pdfmetrics import stringWidth, getFont
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection, PropHolder
from reportlab.graphics.shapes import Drawing, Group, String, Rect, Line, STATE_DEFAULTS
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.utils import isSeq, find_locals, isStr, asNative
from reportlab.graphics.shapes import _baseGFontName
def _getWidths(i, s, fontName, fontSize, subCols):
    S = []
    aS = S.append
    if isSeq(s):
        for j, t in enumerate(s):
            sc = subCols[j, i]
            fN = getattr(sc, 'fontName', fontName)
            fS = getattr(sc, 'fontSize', fontSize)
            m = [stringWidth(x, fN, fS) for x in t.split('\n')]
            m = max(sc.minWidth, m and max(m) or 0)
            aS(m)
            aS(sc.rpad)
        del S[-1]
    else:
        sc = subCols[0, i]
        fN = getattr(sc, 'fontName', fontName)
        fS = getattr(sc, 'fontSize', fontSize)
        m = [stringWidth(x, fN, fS) for x in s.split('\n')]
        aS(max(sc.minWidth, m and max(m) or 0))
    return S