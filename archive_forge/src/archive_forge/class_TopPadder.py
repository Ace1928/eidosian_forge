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
class TopPadder(Flowable):
    """wrap a single flowable so that its first bit will be
    padded to fill out the space so that it appears at the
    bottom of its frame"""

    def __init__(self, f):
        self.__dict__['_TopPadder__f'] = f

    def wrap(self, aW, aH):
        w, h = self.__f.wrap(aW, aH)
        self.__dict__['_TopPadder__dh'] = aH - h
        return (w, h)

    def split(self, aW, aH):
        S = self.__f.split(aW, aH)
        if len(S) > 1:
            S[0] = TopPadder(S[0])
        return S

    def drawOn(self, canvas, x, y, _sW=0):
        self.__f.drawOn(canvas, x, y - max(0, self.__dh - 1e-08), _sW)

    def __setattr__(self, a, v):
        setattr(self.__f, a, v)

    def __getattr__(self, a):
        return getattr(self.__f, a)

    def __delattr__(self, a):
        delattr(self.__f, a)