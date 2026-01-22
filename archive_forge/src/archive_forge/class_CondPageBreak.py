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
class CondPageBreak(Spacer):
    locChanger = 1
    'use up a frame if not enough vertical space effectively CondFrameBreak'

    def __init__(self, height):
        self.height = height

    def __repr__(self):
        return 'CondPageBreak(%s)' % (self.height,)

    def wrap(self, availWidth, availHeight):
        if availHeight < self.height:
            f = self._doctemplateAttr('frame')
            if not f:
                return (availWidth, availHeight)
            from reportlab.platypus.doctemplate import FrameBreak
            f.add_generated_content(FrameBreak)
        return (0, 0)

    def identity(self, maxLen=None):
        return repr(self).replace(')', ',frame=%s)' % self._frameName())