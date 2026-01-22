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
def _splitWord(w, lineWidth, maxWidths, lineno, fontName, fontSize, encoding='utf8'):
    """
    split w into words that fit in lines of length
    maxWidth
    maxWidths[lineno+1]
    .....
    maxWidths[lineno+n]

    then push those new words onto words
    """
    R = []
    aR = R.append
    maxlineno = len(maxWidths) - 1
    wordText = u''
    maxWidth = maxWidths[min(maxlineno, lineno)]
    if isBytes(w):
        w = w.decode(encoding)
    for c in w:
        cw = stringWidth(c, fontName, fontSize, encoding)
        newLineWidth = lineWidth + cw
        if newLineWidth > maxWidth:
            aR(_SplitWord(wordText))
            lineno += 1
            maxWidth = maxWidths[min(maxlineno, lineno)]
            newLineWidth = cw
            wordText = u''
        wordText += c
        lineWidth = newLineWidth
    aR(_SplitWordEnd(wordText))
    return R