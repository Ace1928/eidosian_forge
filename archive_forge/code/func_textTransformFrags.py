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
def textTransformFrags(frags, style):
    tt = style.textTransform
    if tt:
        tt = tt.lower()
        if tt == 'lowercase':
            tt = str.lower
        elif tt == 'uppercase':
            tt = str.upper
        elif tt == 'capitalize':
            tt = str.title
        elif tt == 'none':
            return
        else:
            raise ValueError('ParaStyle.textTransform value %r is invalid' % style.textTransform)
        n = len(frags)
        if n == 1:
            frags[0].text = tt(frags[0].text)
        elif tt is str.title:
            pb = True
            for f in frags:
                u = f.text
                if not u:
                    continue
                if u.startswith(u' ') or pb:
                    u = tt(u)
                else:
                    i = u.find(u' ')
                    if i >= 0:
                        u = u[:i] + tt(u[i:])
                pb = u.endswith(u' ')
                f.text = u
        else:
            for f in frags:
                u = f.text
                if not u:
                    continue
                f.text = tt(u)