from reportlab.platypus.flowables import *
from reportlab.platypus.flowables import _ContainerSpace
from reportlab.lib.units import inch
from reportlab.platypus.paragraph import Paragraph
from reportlab.platypus.frames import Frame
from reportlab.rl_config import defaultPageSize, verbose
import reportlab.lib.sequencer
from reportlab.pdfgen import canvas
from reportlab.lib.utils import isSeq, encode_label, decode_label, annotateException, strTypes
import sys
import logging
class _FrameBreak(LCActionFlowable):
    """
    A special ActionFlowable that allows setting doc._nextFrameIndex

    eg story.append(FrameBreak('mySpecialFrame'))
    """

    def __call__(self, ix=None, resume=0):
        r = self.__class__(self.action + (resume,))
        r._ix = ix
        return r

    def apply(self, doc):
        if getattr(self, '_ix', None):
            doc.handle_nextFrame(self._ix)
        ActionFlowable.apply(self, doc)