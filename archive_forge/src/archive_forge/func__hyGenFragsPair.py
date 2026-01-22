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
def _hyGenFragsPair(hyphenator, FW, newWidth, maxWidth, uriWasteReduce, embeddedHyphenation, hymwl):
    X = _fragWordSplitRep(FW)
    if not X:
        return
    s, X = X
    if isBytes(s):
        s = s.decode('utf8')
    m = _hy_pfx_pat.match(s)
    if m:
        pfx = m.group(0)
        s = s[len(pfx):]
    else:
        pfx = u''
    if isinstance(FW, _SplitFragLL) and FW[-1][1][-1] == '-':
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
    ww = FW[0]
    w0 = newWidth - ww
    R = _uri_split_pairs(s)
    if R is not None:
        if ww > maxWidth or (uriWasteReduce and w0 <= (1 - uriWasteReduce) * maxWidth):
            for h, t in R:
                h = pfx + h
                pos = len(h)
                fx, cc = X[pos]
                FL = FW[1:fx]
                ffx, sfx = FW[fx]
                sfxl = sfx[:pos - cc]
                if sfxl:
                    FL.append((ffx, sfxl))
                sfxr = sfx[pos - cc:]
                FR = FW[fx + 1:]
                if sfxr:
                    FR.insert(0, (ffx, sfxr))
                h = _rebuildFragWord(FL)
                if w0 + h[0] <= maxWidth:
                    return (u'', h, _rebuildFragWord(FR))
        return
    H = _hy_shy_pat.split(s)
    if hyphenator and (_hy_letters_pat.match(s) or (_hy_shy_letters_pat.match(s) and u'' not in H)):
        for h, t in hyphenator(s):
            h = pfx + h
            pos = len(h)
            fx, cc = X[pos]
            FL = FW[1:fx]
            ffx, sfx = FW[fx]
            sfxl = sfx[:pos - cc]
            if not _hy_shy_pat.match(h[-1]):
                jc = u'-'
            else:
                jc = u''
            if sfxl or jc:
                FL.append((ffx, sfxl + jc))
            sfxr = sfx[pos - cc:]
            FR = FW[fx + 1:]
            if sfxr:
                FR.insert(0, (ffx, sfxr))
            h = _rebuildFragWord(FL)
            if w0 + h[0] <= maxWidth:
                return (jc, h, _rebuildFragWord(FR))
    n = len(H)
    if n >= 3 and embeddedHyphenation and (u'' not in H) and _hy_shy_letters_pat.match(s):
        for i in reversed(range(2, n, 2)):
            pos = len(pfx + u''.join(H[:i]))
            fx, cc = X[pos]
            FL = FW[1:fx]
            ffx, sfx = FW[fx]
            sfxl = sfx[:pos - cc]
            if sfxl:
                FL.append((ffx, sfxl))
            sfxr = sfx[pos - cc:]
            FR = FW[fx + 1:]
            if sfxr:
                FR.insert(0, (ffx, sfxr))
            h = _rebuildFragWord(FL)
            if w0 + h[0] <= maxWidth:
                return (u'', h, _rebuildFragWord(FR))