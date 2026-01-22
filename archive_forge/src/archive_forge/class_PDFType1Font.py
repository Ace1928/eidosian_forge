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
class PDFType1Font(PDFObject):
    """no init: set attributes explicitly"""
    __RefOnly__ = 1
    name_attributes = 'Type Subtype BaseFont Name'.split()
    Type = 'Font'
    Subtype = 'Type1'
    local_attributes = 'FirstChar LastChar Widths Encoding ToUnicode FontDescriptor'.split()

    def format(self, document):
        D = {}
        for name in self.name_attributes:
            if hasattr(self, name):
                value = getattr(self, name)
                D[name] = PDFName(value)
        for name in self.local_attributes:
            if hasattr(self, name):
                value = getattr(self, name)
                D[name] = value
        PD = PDFDictionary(D)
        return PD.format(document)