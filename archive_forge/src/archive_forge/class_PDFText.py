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
class PDFText(PDFObject):

    def __init__(self, t, enc='utf-8'):
        self.t = t
        self.enc = enc

    def format(self, document):
        t = self.t
        if isUnicode(t):
            t = t.encode(self.enc)
        result = binascii.hexlify(document.encrypt.encode(t))
        return b'<' + result + b'>'

    def __str__(self):
        dummydoc = DummyDoc()
        return self.format(dummydoc)