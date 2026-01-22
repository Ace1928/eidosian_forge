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
class ImageAndFlowables(_Container, _FindSplitterMixin, Flowable):
    """combine a list of flowables and an Image"""

    def __init__(self, I, F, imageLeftPadding=0, imageRightPadding=3, imageTopPadding=0, imageBottomPadding=3, imageSide='right', imageHref=None):
        self._content = _flowableSublist(F)
        self._I = I
        self._irpad = imageRightPadding
        self._ilpad = imageLeftPadding
        self._ibpad = imageBottomPadding
        self._itpad = imageTopPadding
        self._side = imageSide
        self.imageHref = imageHref

    def deepcopy(self):
        c = copy(self)
        self._reset()
        c.copyContent()
        return c

    def getSpaceAfter(self):
        if hasattr(self, '_C1'):
            C = self._C1
        elif hasattr(self, '_C0'):
            C = self._C0
        else:
            C = self._content
        return _Container.getSpaceAfter(self, C)

    def getSpaceBefore(self):
        return max(self._I.getSpaceBefore(), _Container.getSpaceBefore(self))

    def _reset(self):
        for a in ('_wrapArgs', '_C0', '_C1'):
            try:
                delattr(self, a)
            except:
                pass

    def wrap(self, availWidth, availHeight):
        canv = self.canv
        I = self._I
        if hasattr(self, '_wrapArgs'):
            if self._wrapArgs == (availWidth, availHeight) and getattr(I, '_oldDrawSize', None) is None:
                return (self.width, self.height)
            self._reset()
            I._unRestrictSize()
        self._wrapArgs = (availWidth, availHeight)
        I.wrap(availWidth, availHeight)
        wI, hI = I._restrictSize(availWidth, availHeight)
        self._wI = wI
        self._hI = hI
        ilpad = self._ilpad
        irpad = self._irpad
        ibpad = self._ibpad
        itpad = self._itpad
        self._iW = iW = availWidth - irpad - wI - ilpad
        aH = itpad + hI + ibpad
        if iW > _FUZZ:
            W, H0, self._C0, self._C1 = self._findSplit(canv, iW, aH)
        else:
            W = availWidth
            H0 = 0
        if W > iW + _FUZZ:
            self._C0 = []
            self._C1 = self._content
        aH = self._aH = max(aH, H0)
        self.width = availWidth
        if not self._C1:
            self.height = aH
        else:
            W1, H1 = _listWrapOn(self._C1, availWidth, canv)
            self.height = aH + H1
        return (self.width, self.height)

    def split(self, availWidth, availHeight):
        if hasattr(self, '_wrapArgs'):
            I = self._I
            if self._wrapArgs != (availWidth, availHeight) or getattr(I, '_oldDrawSize', None) is not None:
                self._reset()
                I._unRestrictSize()
        W, H = self.wrap(availWidth, availHeight)
        if self._aH > availHeight:
            return []
        C1 = self._C1
        if C1:
            S = C1[0].split(availWidth, availHeight - self._aH)
            if not S:
                _C1 = []
            else:
                _C1 = [S[0]]
                C1 = S[1:] + C1[1:]
        else:
            _C1 = []
        return [ImageAndFlowables(self._I, self._C0 + _C1, imageLeftPadding=self._ilpad, imageRightPadding=self._irpad, imageTopPadding=self._itpad, imageBottomPadding=self._ibpad, imageSide=self._side, imageHref=self.imageHref)] + C1

    def drawOn(self, canv, x, y, _sW=0):
        if self._side == 'left':
            Ix = x + self._ilpad
            Fx = Ix + self._irpad + self._wI
        else:
            Ix = x + self.width - self._wI - self._irpad
            Fx = x
        self._I.drawOn(canv, Ix, y + self.height - self._itpad - self._hI)
        if self.imageHref:
            canv.linkURL(self.imageHref, (Ix, y + self.height - self._itpad - self._hI, Ix + self._wI, y + self.height), relative=1)
        if self._C0:
            _Container.drawOn(self, canv, Fx, y, content=self._C0, aW=self._iW)
        if self._C1:
            aW, aH = self._wrapArgs
            _Container.drawOn(self, canv, x, y - self._aH, content=self._C1, aW=aW)