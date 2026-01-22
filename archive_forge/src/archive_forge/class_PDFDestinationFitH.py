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
class PDFDestinationFitH(PDFObject):
    typename = 'FitH'

    def __init__(self, page, top):
        self.page = page
        self.top = top

    def format(self, document):
        pageref = document.Reference(self.page)
        A = PDFArray([pageref, PDFName(self.typename), self.top])
        return format(A, document)