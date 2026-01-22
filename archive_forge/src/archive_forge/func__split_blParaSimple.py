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
def _split_blParaSimple(blPara, start, stop):
    f = blPara.clone()
    for a in ('lines', 'kind', 'text'):
        if hasattr(f, a):
            delattr(f, a)
    f.words = list(_yieldBLParaWords(blPara, start, stop))
    if isinstance(f.words[-1], _SplitWordHY):
        f.words[-1].__class__ = _SHYSplit if isinstance(f.words[-1], _SHYStr) else _SplitWordLL
    return [f]