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
def _samePT(self, npt):
    if isSeq(npt):
        return getattr(self, '_nextPageTemplateCycle', [])
    if isinstance(npt, strTypes):
        return npt == (self.pageTemplates[self._nextPageTemplateIndex].id if hasattr(self, '_nextPageTemplateIndex') else self.pageTemplate.id)
    if isinstance(npt, int) and 0 <= npt < len(self.pageTemplates):
        if hasattr(self, '_nextPageTemplateIndex'):
            return npt == self._nextPageTemplateIndex
        return npt == self.pageTemplates.find(self.pageTemplate)