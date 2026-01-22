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
def _hyGenPair(hyphenator, s, ww, newWidth, maxWidth, fontName, fontSize, uriWasteReduce, embeddedHyphenation, hymwl):
    if isBytes(s):
        s = s.decode('utf8')
    m = _hy_pfx_pat.match(s)
    if m:
        pfx = m.group(0)
        s = s[len(pfx):]
    else:
        pfx = u''
    if isinstance(s, _SplitWordLL) and s[-1] == '-':
        sfx = u'-'
        s = s[:-1]
    else:
        m = _hy_sfx_pat.search(s)
        if m:
            sfx = m.group(0)
            s = s[:-len(sfx)]
        else:
            sfx = u''
    if len(s) < hymwl:
        return
    w0 = newWidth - ww
    R = _uri_split_pairs(s)
    if R is not None:
        if ww > maxWidth or (uriWasteReduce and w0 <= (1 - uriWasteReduce) * maxWidth):
            for h, t in R:
                h = pfx + h
                t = t + sfx
                hw = stringWidth(h, fontName, fontSize)
                tw = w0 + hw
                if tw <= maxWidth:
                    return (u'', 0, hw, ww - hw, h, t)
        return
    H = _hy_shy_pat.split(s)
    if hyphenator and (_hy_letters_pat.match(s) or (_hy_shy_letters_pat.match(s) and u'' not in H)):
        hylen = stringWidth(u'-', fontName, fontSize)
        for h, t in hyphenator(s):
            h = pfx + h
            if not _hy_shy_pat.match(h[-1]):
                jc = u'-'
                jclen = hylen
            else:
                jc = u''
                jclen = 0
            t = t + sfx
            hw = stringWidth(h, fontName, fontSize)
            tw = hw + w0 + jclen
            if tw <= maxWidth:
                return (jc, jclen, hw, ww - hw, h, t)
    n = len(H)
    if n >= 3 and embeddedHyphenation and (u'' not in H) and _hy_shy_letters_pat.match(s):
        for i in reversed(range(2, n, 2)):
            h = pfx + ''.join(H[:i])
            t = ''.join(H[i:]) + sfx
            hw = stringWidth(h, fontName, fontSize)
            tw = hw + w0
            if tw <= maxWidth:
                return (u'', 0, hw, ww - hw, h, t)