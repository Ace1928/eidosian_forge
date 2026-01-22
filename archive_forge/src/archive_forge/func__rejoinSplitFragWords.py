from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
def _rejoinSplitFragWords(F):
    """F should be a list of _SplitFrags"""
    R = [0]
    aR = R.append
    wLen = 0
    psty = None
    for f in F:
        wLen += f[0]
        rmhy = isinstance(f, _SplitFragHY)
        for ff in f[1:]:
            sty, t = ff
            if rmhy and ff is f[-1]:
                wLen -= stringWidth(t[-1], sty.fontName, sty.fontSize) + 1e-08
                t = _shyUnsplit(t)
            if psty is sty:
                R[-1] = (sty, _shyUnsplit(R[-1][1], t))
            else:
                aR((sty, t))
                psty = sty
    R[0] = wLen
    return _reconstructSplitFrags(f)(R)