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
class KeepTogether(_ContainerSpace, Flowable):
    splitAtTop = False

    def __init__(self, flowables, maxHeight=None):
        if not hasattr(KeepTogether, 'NullActionFlowable'):
            from reportlab.platypus.doctemplate import NullActionFlowable
            from reportlab.platypus.doctemplate import FrameBreak
            KeepTogether.NullActionFlowable = NullActionFlowable
            KeepTogether.FrameBreak = FrameBreak
        if not flowables:
            flowables = [self.NullActionFlowable()]
        self._content = _flowableSublist(flowables)
        self._maxHeight = maxHeight

    def __repr__(self):
        f = self._content
        L = list(map(repr, f))
        L = '\n' + '\n'.join(L)
        L = L.replace('\n', '\n  ')
        return '%s(%s,maxHeight=%s)' % (self.__class__.__name__, L, self._maxHeight)

    def wrap(self, aW, aH):
        dims = []
        try:
            W, H = _listWrapOn(self._content, aW, self.canv, dims=dims)
        except:
            annotateException('\nraised by class %s(%s)@0x%8.8x wrap\n' % (self.__class__.__name__, self.__class__.__module__, id(self)))
        self._H = H
        self._H0 = dims and dims[0][1] or 0
        self._wrapInfo = (aW, aH)
        return (W, 16777215)

    def split(self, aW, aH):
        if getattr(self, '_wrapInfo', None) != (aW, aH):
            self.wrap(aW, aH)
        S = self._content[:]
        cf = atTop = getattr(self, '_frame', None)
        if cf:
            atTop = getattr(cf, '_atTop', None)
            cAW = cf._width
            cAH = cf._height
        C0 = self._H > aH and (not self._maxHeight or aH > self._maxHeight)
        C1 = self._H0 > aH or (C0 and atTop)
        if C0 or C1:
            fb = False
            panf = self._doctemplateAttr('_peekNextFrame')
            if cf and panf:
                nf = panf()
                nAW = nf._width
                nAH = nf._height
            if C0 and (not (self.splitAtTop and atTop)):
                fb = not (atTop and cf and nf and (cAW >= nAW) and (cAH >= nAH))
            elif nf and nAW >= cf._width and (nAH >= self._H):
                fb = True
            S.insert(0, (self.FrameBreak if fb else self.NullActionFlowable)())
        return S

    def identity(self, maxLen=None):
        msg = '<%s at %s%s> containing :%s' % (self.__class__.__name__, hex(id(self)), self._frameName(), '\n'.join([f.identity() for f in self._content]))
        if maxLen:
            return msg[0:maxLen]
        else:
            return msg