import binascii, codecs, zlib
from collections import OrderedDict
from reportlab.pdfbase import pdfutils
from reportlab import rl_config
from reportlab.lib.utils import open_for_read, makeFileName, isSeq, isBytes, isUnicode, _digester, isStr, bytestr, annotateException, TimeStamp
from reportlab.lib.rl_accel import escapePDF, fp_str, asciiBase85Encode, asciiBase85Decode
from reportlab.pdfbase import pdfmetrics
from hashlib import md5
from sys import stderr
import re
class Destination(PDFObject):
    """

    not a PDFObject!  This is a placeholder that can delegates
    to a pdf object only after it has been defined by the methods
    below.

    EG a Destination can refer to Appendix A before it has been
    defined, but only if Appendix A is explicitly noted as a destination
    and resolved before the document is generated...

    For example the following sequence causes resolution before doc generation.
        d = Destination()
        d.fit() # or other format defining method call
        d.setPage(p)
        (at present setPageRef is called on generation of the page).
    """
    representation = format = page = None

    def __init__(self, name):
        self.name = name
        self.fmt = self.page = None

    def format(self, document):
        f = self.fmt
        if f is None:
            raise ValueError("format not resolved, probably missing URL scheme or undefined destination target for '%s'" % self.name)
        p = self.page
        if p is None:
            raise ValueError("Page not bound, probably missing URL scheme or undefined destination target for '%s'" % self.name)
        f.page = p
        return f.format(document)

    def xyz(self, left, top, zoom):
        self.fmt = PDFDestinationXYZ(None, left, top, zoom)

    def fit(self):
        self.fmt = PDFDestinationFit(None)

    def fitb(self):
        self.fmt = PDFDestinationFitB(None)

    def fith(self, top):
        self.fmt = PDFDestinationFitH(None, top)

    def fitv(self, left):
        self.fmt = PDFDestinationFitV(None, left)

    def fitbh(self, top):
        self.fmt = PDFDestinationFitBH(None, top)

    def fitbv(self, left):
        self.fmt = PDFDestinationFitBV(None, left)

    def fitr(self, left, bottom, right, top):
        self.fmt = PDFDestinationFitR(None, left, bottom, right, top)

    def setPage(self, page):
        self.page = page