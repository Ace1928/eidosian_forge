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
class PDFExponentialFunction(PDFFunction):
    defaults = PDFFunction.defaults + [('Domain', PDFArrayCompact((0.0, 1.0)))]
    required = PDFFunction.required + ('N',)
    permitted = PDFFunction.permitted + ('C0', 'C1', 'N')

    def __init__(self, C0, C1, N, **kw):
        self.C0 = C0
        self.C1 = C1
        self.N = N
        self.otherkw = kw

    def Dict(self, document):
        d = {}
        d.update(self.otherkw)
        d['FunctionType'] = 2
        d['C0'] = PDFArrayCompact(self.C0)
        d['C1'] = PDFArrayCompact(self.C1)
        d['N'] = self.N
        return self.FunctionDict(**d)