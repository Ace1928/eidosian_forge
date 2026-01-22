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
class ParagraphAndImage(Flowable):
    """combine a Paragraph and an Image"""

    def __init__(self, P, I, xpad=3, ypad=3, side='right'):
        self.P = P
        self.I = I
        self.xpad = xpad
        self.ypad = ypad
        self._side = side

    def getSpaceBefore(self):
        return max(self.P.getSpaceBefore(), self.I.getSpaceBefore())

    def getSpaceAfter(self):
        return max(self.P.getSpaceAfter(), self.I.getSpaceAfter())

    def wrap(self, availWidth, availHeight):
        wI, hI = self.I.wrap(availWidth, availHeight)
        self.wI = wI
        self.hI = hI
        self.width = availWidth
        P = self.P
        style = P.style
        xpad = self.xpad
        ypad = self.ypad
        leading = style.leading
        leftIndent = style.leftIndent
        later_widths = availWidth - leftIndent - style.rightIndent
        intermediate_widths = later_widths - xpad - wI
        first_line_width = intermediate_widths - style.firstLineIndent
        P.width = 0
        nIW = int((hI + ypad) / (leading * 1.0))
        P.blPara = P.breakLines([first_line_width] + nIW * [intermediate_widths] + [later_widths])
        if self._side == 'left':
            self._offsets = [wI + xpad] * (1 + nIW) + [0]
        P.height = len(P.blPara.lines) * leading
        self.height = max(hI, P.height)
        return (self.width, self.height)

    def split(self, availWidth, availHeight):
        P, wI, hI, ypad = (self.P, self.wI, self.hI, self.ypad)
        if hI + ypad > availHeight or len(P.frags) <= 0:
            return []
        S = P.split(availWidth, availHeight)
        if not S:
            return S
        P = self.P = S[0]
        del S[0]
        style = P.style
        P.height = len(self.P.blPara.lines) * style.leading
        self.height = max(hI, P.height)
        return [self] + S

    def draw(self):
        canv = self.canv
        if self._side == 'left':
            self.I.drawOn(canv, 0, self.height - self.hI - self.ypad)
            self.P._offsets = self._offsets
            try:
                self.P.drawOn(canv, 0, 0)
            finally:
                del self.P._offsets
        else:
            self.I.drawOn(canv, self.width - self.wI - self.xpad, self.height - self.hI - self.ypad)
            self.P.drawOn(canv, 0, 0)