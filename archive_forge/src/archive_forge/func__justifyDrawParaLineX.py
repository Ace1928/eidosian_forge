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
def _justifyDrawParaLineX(tx, offset, line, last=0):
    tx._x_offset = offset
    setXPos(tx, offset)
    extraSpace = line.extraSpace
    simple = line.lineBreak or -1e-08 < extraSpace <= 1e-08 or (last and extraSpace > -1e-08)
    if not simple:
        nSpaces = line.wordCount + sum([_nbspCount(w.text) for w in line.words if not hasattr(w, 'cbDefn')]) - 1
        simple = nSpaces <= 0
    if not simple:
        tx.setWordSpace(extraSpace / float(nSpaces))
        _putFragLine(offset, tx, line, last, 'justify')
        tx.setWordSpace(0)
    else:
        _putFragLine(offset, tx, line, last, 'justify')
    setXPos(tx, -offset)