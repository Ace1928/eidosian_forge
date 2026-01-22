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
class PageTemplate:
    """
    essentially a list of Frames and an onPage routine to call at the start
    of a page when this is selected. onPageEnd gets called at the end.
    derived classes can also implement beforeDrawPage and afterDrawPage if they want
    """

    def __init__(self, id=None, frames=[], onPage=_doNothing, onPageEnd=_doNothing, pagesize=None, autoNextPageTemplate=None, cropBox=None, artBox=None, trimBox=None, bleedBox=None):
        frames = frames or []
        if not isSeq(frames):
            frames = [frames]
        assert [x for x in frames if not isinstance(x, Frame)] == [], 'frames argument error'
        self.id = id
        self.frames = frames
        self.onPage = onPage
        self.onPageEnd = onPageEnd
        self.pagesize = pagesize
        self.autoNextPageTemplate = autoNextPageTemplate
        self.cropBox = cropBox
        self.artBox = artBox
        self.trimBox = trimBox
        self.bleedBox = bleedBox

    def beforeDrawPage(self, canv, doc):
        """Override this if you want additional functionality or prefer
        a class based page routine.  Called before any flowables for
        this page are processed."""
        pass

    def checkPageSize(self, canv, doc):
        """This gets called by the template framework
        If canv size != template size then the canv size is set to
        the template size or if that's not available to the
        doc size.
        """
        cp = None
        dp = None
        sp = None
        if canv._pagesize:
            cp = list(map(int, canv._pagesize))
        if self.pagesize:
            sp = list(map(int, self.pagesize))
        if doc.pagesize:
            dp = list(map(int, doc.pagesize))
        if cp != sp:
            if sp:
                canv.setPageSize(self.pagesize)
            elif cp != dp:
                canv.setPageSize(doc.pagesize)
        for box in ('crop', 'art', 'trim', 'bleed'):
            size = getattr(self, box + 'Box', None)
            if size:
                canv.setCropBox(size, name=box)

    def afterDrawPage(self, canv, doc):
        """This is called after the last flowable for the page has
        been processed.  You might use this if the page header or
        footer needed knowledge of what flowables were drawn on
        this page."""
        pass