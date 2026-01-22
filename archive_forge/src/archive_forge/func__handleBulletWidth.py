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
def _handleBulletWidth(bulletText, style, maxWidths):
    """work out bullet width and adjust maxWidths[0] if neccessary
    """
    if bulletText:
        if isStr(bulletText):
            bulletWidth = stringWidth(bulletText, style.bulletFontName, style.bulletFontSize)
        else:
            bulletWidth = 0
            for f in bulletText:
                bulletWidth += stringWidth(f.text, f.fontName, f.fontSize)
        bulletLen = style.bulletIndent + bulletWidth + 0.6 * style.bulletFontSize
        if style.wordWrap == 'RTL':
            indent = style.rightIndent + style.firstLineIndent
        else:
            indent = style.leftIndent + style.firstLineIndent
        if bulletLen > indent:
            maxWidths[0] -= bulletLen - indent