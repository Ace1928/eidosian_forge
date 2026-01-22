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
class NotAtTopPageBreak(FrameActionFlowable):
    locChanger = 1

    def __init__(self, nextTemplate=None):
        self.nextTemplate = nextTemplate

    def frameAction(self, frame):
        if not frame._atTop:
            frame.add_generated_content(PageBreak(nextTemplate=self.nextTemplate))