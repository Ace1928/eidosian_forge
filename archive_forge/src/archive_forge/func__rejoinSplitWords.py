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
def _rejoinSplitWords(R):
    """R can be a list of pure _SplitWord or _SHYStr"""
    if isinstance(R[0], _SHYStr):
        r = R[0]
        for _ in R[:]:
            r = _shyUnsplit(r, _)
        return r
    elif isinstance(R[0], _SplitWordHY):
        cf = str if isinstance(R[-1], _SplitWordEnd) else _SplitWordHY
        s = u''.join((_[:-1] if isinstance(_, _SplitWordHY) else _ for _ in R))
        return s if isinstance(R[-1], _SplitWordEnd) else _SplitWordHY(s + u'-')
    else:
        return ''.join(R)