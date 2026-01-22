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
class PageBreak(UseUpSpace):
    locChanger = 1
    'Move on to the next page in the document.\n       This works by consuming all remaining space in the frame!'

    def __init__(self, nextTemplate=None):
        self.nextTemplate = nextTemplate

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, repr(self.nextTemplate) if self.nextTemplate else '')