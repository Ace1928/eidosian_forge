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
class DDIndenter(Flowable):
    _IndenterAttrs = '_flowable _leftIndent _rightIndent width height'.split()

    def __init__(self, flowable, leftIndent=0, rightIndent=0):
        self._flowable = flowable
        self._leftIndent = leftIndent
        self._rightIndent = rightIndent
        self.width = None
        self.height = None

    def split(self, aW, aH):
        S = self._flowable.split(aW - self._leftIndent - self._rightIndent, aH)
        return [DDIndenter(s, leftIndent=self._leftIndent, rightIndent=self._rightIndent) for s in S]

    def drawOn(self, canv, x, y, _sW=0):
        self._flowable.drawOn(canv, x + self._leftIndent, y, max(0, _sW - self._leftIndent - self._rightIndent))

    def wrap(self, aW, aH):
        w, h = self._flowable.wrap(aW - self._leftIndent - self._rightIndent, aH)
        self.width = w + self._leftIndent + self._rightIndent
        self.height = h
        return (self.width, h)

    def __getattr__(self, a):
        if a in self._IndenterAttrs:
            try:
                return self.__dict__[a]
            except KeyError:
                if a not in ('spaceBefore', 'spaceAfter'):
                    raise AttributeError(f'{self!r} has no attribute {a} dict={self.__dict__}')
        return getattr(self._flowable, a)

    def __setattr__(self, a, v):
        if a in self._IndenterAttrs:
            self.__dict__[a] = v
        else:
            setattr(self._flowable, a, v)

    def __delattr__(self, a):
        if a in self._IndenterAttrs:
            del self.__dict__[a]
        else:
            delattr(self._flowable, a)

    def identity(self, maxLen=None):
        return '%s containing %s' % (self.__class__.__name__, self._flowable.identity(maxLen))