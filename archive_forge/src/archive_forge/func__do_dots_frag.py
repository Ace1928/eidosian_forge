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
def _do_dots_frag(cur_x, cur_x_s, maxWidth, xs, tx, left=True):
    text, fontName, fontSize, textColor, backColor, dy = _getDotsInfo(xs.style)
    txtlen = tx._canvas.stringWidth(text, fontName, fontSize)
    if cur_x_s + txtlen <= maxWidth:
        if tx._fontname != fontName or tx._fontsize != fontSize:
            tx.setFont(fontName, fontSize)
        if left:
            maxWidth += getattr(tx, '_dotsOffsetX', tx._x0)
        tx.setTextOrigin(0, xs.cur_y + dy)
        setXPos(tx, cur_x_s - cur_x)
        n = int((maxWidth - cur_x_s) / txtlen)
        setXPos(tx, maxWidth - txtlen * n)
        if xs.textColor != textColor:
            tx.setFillColor(textColor)
        if backColor:
            xs.backColors.append((cur_x, maxWidth, backColor))
        tx._textOut(n * text, 1)
        if dy:
            tx.setTextOrigin(tx._x0, xs.cur_y - dy)