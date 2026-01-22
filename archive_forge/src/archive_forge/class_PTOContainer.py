import os
from copy import deepcopy, copy
from reportlab.lib.colors import gray, lightgrey
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import _baseFontName
from reportlab.lib.utils import strTypes, rl_safe_exec, annotateException
from reportlab.lib.abag import ABag
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.rl_config import _FUZZ, overlapAttachedSpace, ignoreContainerActions, listWrapOnFakeWidth
from reportlab.lib.sequencer import _type2formatter
from reportlab.lib.styles import ListStyle
class PTOContainer(_Container, Flowable):
    """PTOContainer(contentList,trailerList,headerList)

    A container for flowables decorated with trailer & header lists.
    If the split operation would be called then the trailer and header
    lists are injected before and after the split. This allows specialist
    "please turn over" and "continued from previous" like behaviours."""

    def __init__(self, content, trailer=None, header=None):
        I = _PTOInfo(trailer, header)
        self._content = C = []
        for _ in _flowableSublist(content):
            if isinstance(_, PTOContainer):
                C.extend(_._content)
            else:
                C.append(_)
                if not hasattr(_, '_ptoinfo'):
                    _._ptoinfo = I

    def wrap(self, availWidth, availHeight):
        self.width, self.height = _listWrapOn(self._content, availWidth, self.canv)
        return (self.width, self.height)

    def split(self, availWidth, availHeight):
        from reportlab.platypus.doctemplate import Indenter
        if availHeight < 0:
            return []
        canv = self.canv
        C = self._content
        x = i = H = pS = hx = 0
        n = len(C)
        I2W = {}
        dLeft = dRight = 0
        for x in range(n):
            c = C[x]
            I = c._ptoinfo
            if I not in I2W.keys():
                T = I.trailer
                Hdr = I.header
                tW, tH = _listWrapOn(T, availWidth, self.canv)
                if len(T):
                    tSB = T[0].getSpaceBefore()
                else:
                    tSB = 0
                I2W[I] = (T, tW, tH, tSB)
            else:
                T, tW, tH, tSB = I2W[I]
            _, h = c.wrapOn(canv, availWidth, 268435455)
            if isinstance(c, Indenter):
                dw = c.left + c.right
                dLeft += c.left
                dRight += c.right
                availWidth -= dw
                pS = 0
                hx = 0
            else:
                if x:
                    hx = max(c.getSpaceBefore() - pS, 0)
                    h += hx
                pS = c.getSpaceAfter()
            H += h + pS
            tHS = tH + max(tSB, pS)
            if H + tHS >= availHeight - _FUZZ:
                break
            i += 1
        H -= h + pS
        aH = (availHeight - H - tHS - hx) * 0.99999
        if aH >= 0.05 * availHeight:
            SS = c.splitOn(canv, availWidth, aH)
        else:
            SS = []
        if abs(dLeft) + abs(dRight) > 1e-08:
            R1I = [Indenter(-dLeft, -dRight)]
            R2I = [Indenter(dLeft, dRight)]
        else:
            R1I = R2I = []
        if not SS:
            j = i
            while i > 1 and C[i - 1].getKeepWithNext():
                i -= 1
                C[i].keepWithNext = 0
            if i == 1 and C[0].getKeepWithNext():
                i = j
                C[0].keepWithNext = 0
        F = [UseUpSpace()]
        if len(SS) > 1:
            R1 = C[:i] + SS[:1] + R1I + T + F
            R2 = Hdr + R2I + SS[1:] + C[i + 1:]
        elif not i:
            return []
        else:
            R1 = C[:i] + R1I + T + F
            R2 = Hdr + R2I + C[i:]
        T = R1 + [PTOContainer(R2, [copy(x) for x in I.trailer], [copy(x) for x in I.header])]
        return T