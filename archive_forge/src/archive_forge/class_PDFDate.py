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
class PDFDate(PDFObject):

    def __init__(self, ts=None, dateFormatter=None):
        if ts is None:
            ts = TimeStamp()
        self._init(ts)
        self.dateFormatter = dateFormatter

    def _init(self, ts):
        self.date = ts.YMDhms
        self.dhh = ts.dhh
        self.dmm = ts.dmm

    def format(self, doc):
        dfmt = self.dateFormatter or (lambda yyyy, mm, dd, hh, m, s: "D:%04d%02d%02d%02d%02d%02d%+03d'%02d'" % (yyyy, mm, dd, hh, m, s, self.dhh, self.dmm))
        return format(PDFString(dfmt(*self.date)), doc)