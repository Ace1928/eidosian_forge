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
def _do_link_line(i, t_off, ws, tx):
    xs = tx.XtraState
    leading = xs.style.leading
    y = xs.cur_y - i * leading - xs.f.fontSize / 8.0
    text = ' '.join(xs.lines[i][1])
    textlen = tx._canvas.stringWidth(text, tx._fontname, tx._fontsize)
    for n, link in xs.link:
        _doLink(tx, link, (t_off, y, t_off + textlen, y + leading))