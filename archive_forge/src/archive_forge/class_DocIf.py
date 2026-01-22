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
class DocIf(DocPara):

    def __init__(self, cond, thenBlock, elseBlock=[]):
        Flowable.__init__(self)
        self.expr = cond
        self.blocks = (elseBlock or [], thenBlock)

    def checkBlock(self, block):
        if not isinstance(block, (list, tuple)):
            block = (block,)
        return block

    def wrap(self, aW, aH):
        self.add_content(*self.checkBlock(self.blocks[int(bool(self.funcWrap(aW, aH)))]))
        return (0, 0)