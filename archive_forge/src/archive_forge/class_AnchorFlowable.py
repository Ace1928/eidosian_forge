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
class AnchorFlowable(Spacer):
    """create a bookmark in the pdf"""
    _ZEROSIZE = 1
    _SPACETRANSFER = True

    def __init__(self, name):
        Spacer.__init__(self, 0, 0)
        self._name = name

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._name)

    def wrap(self, aW, aH):
        return (0, 0)

    def draw(self):
        self.canv.bookmarkHorizontal(self._name, 0, 0)