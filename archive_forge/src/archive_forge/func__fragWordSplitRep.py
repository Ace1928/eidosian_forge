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
def _fragWordSplitRep(FW):
    """takes a frag word and assembles a unicode word from it
    if a rise is seen or a non-zerowidth cbdefn then we return
    None. Otherwise we return (uword,([i1,c1],[i2,c2],...])
    where each ii is the index of the word fragment in the word
    """
    cc = plen = 0
    X = []
    eX = X.extend
    U = []
    aU = U.append
    for i in range(1, len(FW)):
        f, t = FW[i]
        if f.rise != 0:
            return None
        if hasattr(f, 'cbDefn') and getattr(f.cbDefn, 'width', 0):
            return
        if not t:
            continue
        if isBytes(t):
            t = t.decode('utf8')
        aU(t)
        eX(len(t) * [(i, cc)])
        cc += len(t)
    return (u''.join(U), tuple(X))